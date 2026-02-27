# -- coding: utf-8 --

import ctypes
import logging
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional, Tuple
import time

import numpy as np

from camera.base import BaseCamera, CameraConfig, CaptureResult, register_camera
from utils.path_time import UtcDailyDirCache, build_dated_frame_path

L = logging.getLogger("vision_runtime.camera.opt")

SCI_CAMERA_OK = 0


class SciCamTLType(IntEnum):
    SciCam_TLType_Gige = 1


class SciCamDeviceXmlType(IntEnum):
    SciCam_DeviceXml_Camera = 0


class SciCamPixelType(IntEnum):
    Mono1p = 0x01010037
    Mono2p = 0x01020038
    Mono4p = 0x01040039
    Mono8s = 0x01080002
    Mono8 = 0x01080001
    Mono10 = 0x01100003
    Mono10p = 0x010A0046
    Mono12 = 0x01100005
    Mono12p = 0x010C0047
    Mono14 = 0x01100025
    Mono16 = 0x01100007
    RGB8 = 0x02180014
    Mono10Packed = 0x010C0004
    Mono12Packed = 0x010C0006
    Mono14p = 0x010E0104


class SciCamPayloadMode(IntEnum):
    SciCam_PayloadMode_2D = 1


TARGET_MONO = SciCamPixelType.Mono8
TARGET_COLOR = SciCamPixelType.RGB8

system = platform.system()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_LOADED_PATH = None


def _load_sdk():
    if system == "Windows":
        env_key = f"OPTMV_COMMON_RUNENV{'64' if platform.architecture()[0] == '64bit' else '32'}"
        sdk_path = os.environ.get(env_key)
        lib_path = (
            os.path.join(sdk_path, "SciCamSDK.dll")
            if sdk_path
            else os.path.join(_SCRIPT_DIR, "SciCamSDK.dll")
        )
    elif system == "Darwin":
        lib_path = os.path.join(_SCRIPT_DIR, "libSciCamSDK.dylib")
    else:
        lib_path = os.path.join(_SCRIPT_DIR, "libSciCamSDK.so")

    if not os.path.exists(lib_path):
        raise EnvironmentError(f"SDK library not found at {lib_path}")
    return ctypes.CDLL(lib_path), lib_path


try:
    SciCamCtrlDll, SDK_LOADED_PATH = _load_sdk()
except OSError as e:
    raise EnvironmentError(f"Failed to load SciCam SDK: {e}")


class _SCI_DEVICE_GIGE_INFO_(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_ubyte),
        ("name", ctypes.c_ubyte * 64),
        ("manufactureName", ctypes.c_ubyte * 32),
        ("modelName", ctypes.c_ubyte * 32),
        ("version", ctypes.c_ubyte * 32),
        ("userDefineName", ctypes.c_ubyte * 16),
        ("serialNumber", ctypes.c_ubyte * 16),
        ("mac", ctypes.c_ubyte * 6),
        ("ip", ctypes.c_uint),
        ("mask", ctypes.c_uint),
        ("gateway", ctypes.c_uint),
        ("adapterIp", ctypes.c_uint),
        ("adapterMask", ctypes.c_uint),
        ("adapterName", ctypes.c_ubyte * 260),
    ]


class _SCI_DEVICE_USB3_INFO_(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_ubyte),
        ("name", ctypes.c_ubyte * 64),
        ("manufactureName", ctypes.c_ubyte * 64),
        ("modelName", ctypes.c_ubyte * 64),
        ("version", ctypes.c_ubyte * 64),
        ("userDefineName", ctypes.c_ubyte * 64),
        ("serialNumber", ctypes.c_ubyte * 64),
        ("guid", ctypes.c_ubyte * 64),
        ("U3VVersion", ctypes.c_ubyte * 64),
        ("GenCPVersion", ctypes.c_ubyte * 64),
    ]


class _SCI_DEVICE_CL_INFO_(ctypes.Structure):
    _fields_ = [
        ("cardStatus", ctypes.c_ubyte),
        ("cardName", ctypes.c_ubyte * 64),
        ("cardManufacture", ctypes.c_ubyte * 64),
        ("cardModel", ctypes.c_ubyte * 64),
        ("cardVersion", ctypes.c_ubyte * 64),
        ("cardUserDefineName", ctypes.c_ubyte * 64),
        ("cardSerialNumber", ctypes.c_ubyte * 64),
        ("cameraStatus", ctypes.c_ubyte),
        ("cameraType", ctypes.c_ubyte),
        ("cameraBaud", ctypes.c_uint),
        ("cameraManufacture", ctypes.c_ubyte * 64),
        ("cameraFamily", ctypes.c_ubyte * 64),
        ("cameraModel", ctypes.c_ubyte * 64),
        ("cameraVersion", ctypes.c_ubyte * 64),
        ("cameraSerialNumber", ctypes.c_ubyte * 64),
        ("cameraSerialPort", ctypes.c_ubyte * 64),
        ("cameraProtocol", ctypes.c_ubyte * 256),
    ]


class _SCI_DEVICE_CXP_INFO_(ctypes.Structure):
    _fields_ = [("extend", ctypes.c_ubyte * 2048)]


class SCI_DEVICE_INFO_INFO_UNION(ctypes.Union):
    _fields_ = [
        ("gigeInfo", _SCI_DEVICE_GIGE_INFO_),
        ("usb3Info", _SCI_DEVICE_USB3_INFO_),
        ("clInfo", _SCI_DEVICE_CL_INFO_),
        ("cxpInfo", _SCI_DEVICE_CXP_INFO_),
    ]


class _SCI_DEVICE_INFO_(ctypes.Structure):
    _fields_ = [
        ("tlType", ctypes.c_int),
        ("devType", ctypes.c_int),
        ("reserve", ctypes.c_ubyte * 256),
        ("info", SCI_DEVICE_INFO_INFO_UNION),
    ]


class _SCI_DEVICE_INFO_LIST_(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint), ("pDevInfo", _SCI_DEVICE_INFO_ * 256)]


class _SCI_CAM_IMAGE_ATTRIBUTE_(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("offsetX", ctypes.c_uint64),
        ("offsetY", ctypes.c_uint64),
        ("paddingX", ctypes.c_uint64),
        ("paddingY", ctypes.c_uint64),
        ("pixelType", ctypes.c_int),
        ("reserve", ctypes.c_ubyte * 32),
    ]


class _SCI_CAM_PAYLOAD_ATTRIBUTE_(ctypes.Structure):
    _fields_ = [
        ("frameID", ctypes.c_uint64),
        ("isComplete", ctypes.c_bool),
        ("hasChunk", ctypes.c_bool),
        ("timeStamp", ctypes.c_uint64),
        ("payloadMode", ctypes.c_int),
        ("imgAttr", _SCI_CAM_IMAGE_ATTRIBUTE_),
        ("reserve", ctypes.c_ubyte * 64),
    ]


SCI_DEVICE_INFO = _SCI_DEVICE_INFO_
SCI_DEVICE_INFO_LIST = _SCI_DEVICE_INFO_LIST_
SCI_CAM_IMAGE_ATTRIBUTE = _SCI_CAM_IMAGE_ATTRIBUTE_
SCI_CAM_PAYLOAD_ATTRIBUTE = _SCI_CAM_PAYLOAD_ATTRIBUTE_

ref = ctypes.byref


def _setup_sdk():
    funcs = {
        "SciCam_DiscoveryDevices": (
            (ctypes.POINTER(SCI_DEVICE_INFO_LIST), ctypes.c_uint),
            ctypes.c_uint,
        ),
        "SciCam_CreateDevice": (
            (ctypes.c_void_p, ctypes.POINTER(SCI_DEVICE_INFO)),
            ctypes.c_uint,
        ),
        "SciCam_DeleteDevice": ((ctypes.c_void_p,), ctypes.c_uint),
        "SciCam_OpenDevice": ((ctypes.c_void_p,), ctypes.c_uint),
        "SciCam_CloseDevice": ((ctypes.c_void_p,), ctypes.c_uint),
        "SciCam_SetGrabTimeout": ((ctypes.c_void_p, ctypes.c_uint), ctypes.c_uint),
        "SciCam_StartGrabbing": ((ctypes.c_void_p,), ctypes.c_uint),
        "SciCam_StopGrabbing": ((ctypes.c_void_p,), ctypes.c_uint),
        "SciCam_Grab": (
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.c_uint,
        ),
        "SciCam_FreePayload": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
        "SciCam_SetCommandValueEx": (
            (ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p),
            ctypes.c_uint,
        ),
        "SciCam_Payload_GetAttribute": (
            (ctypes.c_void_p, ctypes.POINTER(SCI_CAM_PAYLOAD_ATTRIBUTE)),
            ctypes.c_uint,
        ),
        "SciCam_Payload_GetImage": (
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.c_uint,
        ),
        "SciCam_Payload_ConvertImage": (
            (
                ctypes.POINTER(SCI_CAM_IMAGE_ATTRIBUTE),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
            ),
            ctypes.c_uint,
        ),
        "SciCam_Payload_SaveImage": (
            (
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
            ),
            ctypes.c_uint,
        ),
    }
    for func_name, (argtypes, restype) in funcs.items():
        func = getattr(SciCamCtrlDll, func_name)
        func.argtypes = argtypes
        func.restype = restype


_setup_sdk()


class SciCamSDK:
    """Small wrapper to keep SDK surface in one place for easier testing/mocking."""

    def __init__(self, dll: ctypes.CDLL):
        self.dll = dll

    def discovery(self):
        dev_infos = SCI_DEVICE_INFO_LIST()
        ret = self.dll.SciCam_DiscoveryDevices(
            ref(dev_infos), ctypes.c_uint(SciCamTLType.SciCam_TLType_Gige)
        )
        return ret, dev_infos

    def create_device(self, handle, info):
        return self.dll.SciCam_CreateDevice(ctypes.byref(handle), ctypes.byref(info))

    def delete_device(self, handle):
        return self.dll.SciCam_DeleteDevice(handle)

    def open_device(self, handle):
        return self.dll.SciCam_OpenDevice(handle)

    def close_device(self, handle):
        return self.dll.SciCam_CloseDevice(handle)

    def set_grab_timeout(self, handle, timeout_ms: int):
        return self.dll.SciCam_SetGrabTimeout(handle, ctypes.c_uint(timeout_ms))

    def start_grabbing(self, handle):
        return self.dll.SciCam_StartGrabbing(handle)

    def stop_grabbing(self, handle):
        return self.dll.SciCam_StopGrabbing(handle)

    def trigger(self, handle):
        return self.dll.SciCam_SetCommandValueEx(
            handle,
            ctypes.c_int(SciCamDeviceXmlType.SciCam_DeviceXml_Camera),
            b"TriggerSoftware",
        )

    def grab_payload(self, handle):
        payload = ctypes.c_void_p()
        ret = self.dll.SciCam_Grab(handle, ref(payload))
        return ret, payload

    def free_payload(self, handle, payload):
        return self.dll.SciCam_FreePayload(handle, payload)

    def convert_payload(self, payload, dst_type):
        """
        Convert payload to (buffer, width, height, dst_type, attr) or return (None, error) on failure.
        Handles attribute lookup, image pointer fetch, and conversion buffer sizing.
        """
        attr = SCI_CAM_PAYLOAD_ATTRIBUTE()
        ret = self.dll.SciCam_Payload_GetAttribute(payload, ref(attr))
        if ret != SCI_CAMERA_OK:
            return None, f"payload_get_attribute(0x{ret:08X})"
        if not (
            attr.isComplete
            and attr.payloadMode == SciCamPayloadMode.SciCam_PayloadMode_2D
        ):
            return None, "payload_incomplete"

        img_ptr = ctypes.c_void_p()
        ret = self.dll.SciCam_Payload_GetImage(payload, ref(img_ptr))
        if ret != SCI_CAMERA_OK or not img_ptr.value:
            return None, f"payload_get_image(0x{ret:08X})"
        size = ctypes.c_int(0)
        ret = self.dll.SciCam_Payload_ConvertImage(
            ref(attr.imgAttr), img_ptr, dst_type, None, ref(size), ctypes.c_bool(True)
        )
        if ret != SCI_CAMERA_OK or size.value <= 0:
            return None, f"convert_size(0x{ret:08X})"

        buf = (ctypes.c_ubyte * size.value)()
        ret = self.dll.SciCam_Payload_ConvertImage(
            ref(attr.imgAttr), img_ptr, dst_type, buf, ref(size), ctypes.c_bool(True)
        )
        if ret != SCI_CAMERA_OK:
            return None, f"convert(0x{ret:08X})"

        width = int(attr.imgAttr.width)
        height = int(attr.imgAttr.height)
        return (buf, width, height, dst_type, attr), None

    def save_image(self, path: str, dst_type, buf, width: int, height: int):
        return self.dll.SciCam_Payload_SaveImage(
            path.encode("utf-8"),
            dst_type,
            buf,
            ctypes.c_int64(width),
            ctypes.c_int64(height),
        )


SDK = SciCamSDK(SciCamCtrlDll)


def _dst_pixel_type(output_pixel_format: str):
    fmt = str(output_pixel_format or "").strip().lower()
    if fmt == "mono8":
        return TARGET_MONO
    return TARGET_COLOR


_DATE_CACHE = UtcDailyDirCache()


def _process_payload(
    payload,
    cfg: CameraConfig,
    frame_id: Optional[int] = None,
    trigger_ts_utc: Optional[datetime] = None,
) -> Tuple[Optional[Tuple[Optional[str], np.ndarray, datetime]], Optional[str]]:
    dst_type = _dst_pixel_type(cfg.output_pixel_format)
    conversion, err = SDK.convert_payload(payload, dst_type)
    if not conversion:
        return None, err
    buf, width, height, dst_type, attr = conversion
    now_utc = datetime.now(timezone.utc)
    save_ret = SCI_CAMERA_OK
    path = None
    if cfg.save_images:
        file_id = frame_id if frame_id is not None else attr.frameID
        path, _ = build_dated_frame_path(
            cfg.save_dir,
            file_id,
            cfg.ext,
            ts_utc=(trigger_ts_utc or now_utc),
            cache=_DATE_CACHE,
        )
        save_ret = SDK.save_image(path, dst_type, buf, width, height)
        if save_ret != SCI_CAMERA_OK:
            path = None

    # Build numpy array for detection/overlay (stay in BGR per contracts)
    arr = np.ctypeslib.as_array(buf)
    if str(cfg.output_pixel_format).lower() == "mono8":
        arr = arr.reshape((height, width))
    else:
        arr = arr.reshape((height, width, 3))  # BGR order from SDK
    return (
        path if (cfg.save_images and save_ret == SCI_CAMERA_OK) else None,
        arr,
        now_utc,
    ), None


def discovery():
    return SDK.discovery()


def _carray_to_str(arr):
    raw = bytes(arr)
    return raw.split(b"\x00", 1)[0].decode(errors="ignore").strip()


def _check(ret, msg):
    if ret != SCI_CAMERA_OK:
        raise RuntimeError(f"{msg} (0x{ret:08X}).")


@register_camera("opt")
class OptSciCamera(BaseCamera):
    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self.handle = ctypes.c_void_p()

    def _trigger(self):
        return SDK.trigger(self.handle)

    def _grab(self):
        return SDK.grab_payload(self.handle)

    def capture_once(self, idx, triggered_at: Optional[datetime] = None):
        last_err = None
        for _ in range(self.cfg.max_retry_per_frame):
            trig_ret = self._trigger()
            if trig_ret != SCI_CAMERA_OK:
                last_err = f"trigger(0x{trig_ret:08X})"
                continue

            step_t = time.perf_counter()
            grab_ret, payload = self._grab()
            if grab_ret != SCI_CAMERA_OK:
                last_err = f"grab(0x{grab_ret:08X})"
                continue
            grab_ms = (time.perf_counter() - step_t) * 1000

            try:
                result, err = _process_payload(
                    payload,
                    self.cfg,
                    frame_id=idx,
                    trigger_ts_utc=triggered_at,
                )
                if err:
                    last_err = err
                    continue
                if result:
                    path, arr, ts = result
                    if path:
                        L.info("[%5s] @ %s", idx, path)
                    else:
                        L.info("[%5s] captured (no file save)", idx)
                    return CaptureResult(
                        success=True,
                        trigger_seq=idx,
                        source="",
                        device_id=str(self.cfg.device_index),
                        image=arr,
                        timings={
                            "grab_ms": grab_ms,
                        },
                        captured_at=ts,
                    )
            finally:
                SDK.free_payload(self.handle, payload)

        if last_err:
            L.error(
                "[%s] failed after %d attempts: %s",
                idx,
                self.cfg.max_retry_per_frame,
                last_err,
            )
        else:
            L.error("[%s] failed after %d attempts", idx, self.cfg.max_retry_per_frame)
        return CaptureResult(
            success=False,
            trigger_seq=idx,
            device_id=str(self.cfg.device_index),
            error=last_err or "capture_failed",
        )

    @contextmanager
    def session(self):
        ret, devs = discovery()
        _check(ret, "Discovery failed")
        if devs.count == 0:
            raise RuntimeError("No GigE cameras found")
        if self.cfg.device_index >= devs.count:
            raise RuntimeError(
                f"Device index {self.cfg.device_index} out of range (found {devs.count})"
            )

        info = devs.pDevInfo[self.cfg.device_index].info.gigeInfo
        L.info("Camera [%d] name=%s", self.cfg.device_index, _carray_to_str(info.name))

        _check(
            SDK.create_device(self.handle, devs.pDevInfo[self.cfg.device_index]),
            "create",
        )
        try:
            _check(SDK.open_device(self.handle), "open")
            _check(
                SDK.set_grab_timeout(self.handle, self.cfg.timeout_ms), "set timeout"
            )
            _check(SDK.start_grabbing(self.handle), "start grabbing")
            yield self
        finally:
            try:
                SDK.stop_grabbing(self.handle)
            finally:
                try:
                    SDK.close_device(self.handle)
                finally:
                    SDK.delete_device(self.handle)


__all__ = ["OptSciCamera", "SDK_LOADED_PATH"]
