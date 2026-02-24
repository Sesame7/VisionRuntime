# -- coding: utf-8 --

import ctypes
import logging
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

from camera.base import BaseCamera, CameraConfig, CaptureResult, register_camera

L = logging.getLogger("vision_runtime.camera.hik")

MV_OK = 0

MV_MAX_DEVICE_NUM = 256

MV_GIGE_DEVICE = 0x00000001
MV_USB_DEVICE = 0x00000004
MV_GENTL_GIGE_DEVICE = 0x00000040
MV_GENTL_CAMERALINK_DEVICE = 0x00000080
MV_GENTL_CXP_DEVICE = 0x00000100
MV_GENTL_XOF_DEVICE = 0x00000200

MV_ACCESS_Exclusive = 1

MV_Image_Bmp = 1
MV_Image_Jpeg = 2
MV_Image_Png = 3

MV_GVSP_PIX_MONO8 = 0x01080001
MV_GVSP_PIX_BGR8_PACKED = 0x02180015

INFO_MAX_BUFFER_SIZE = 64


def _mv_gvsp_pixel_type():
    return ctypes.c_uint if platform.system() == "Windows" else ctypes.c_int64


MvGvspPixelType = _mv_gvsp_pixel_type()
MV_SAVE_IMAGE_TYPE = ctypes.c_int


class MV_GIGE_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("nIpCfgOption", ctypes.c_uint),
        ("nIpCfgCurrent", ctypes.c_uint),
        ("nCurrentIp", ctypes.c_uint),
        ("nCurrentSubNetMask", ctypes.c_uint),
        ("nDefultGateWay", ctypes.c_uint),
        ("chManufacturerName", ctypes.c_ubyte * 32),
        ("chModelName", ctypes.c_ubyte * 32),
        ("chDeviceVersion", ctypes.c_ubyte * 32),
        ("chManufacturerSpecificInfo", ctypes.c_ubyte * 48),
        ("chSerialNumber", ctypes.c_ubyte * 16),
        ("chUserDefinedName", ctypes.c_ubyte * 16),
        ("nNetExport", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
    ]


class MV_USB3_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("CrtlInEndPoint", ctypes.c_ubyte),
        ("CrtlOutEndPoint", ctypes.c_ubyte),
        ("StreamEndPoint", ctypes.c_ubyte),
        ("EventEndPoint", ctypes.c_ubyte),
        ("idVendor", ctypes.c_ushort),
        ("idProduct", ctypes.c_ushort),
        ("nDeviceNumber", ctypes.c_uint),
        ("chDeviceGUID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chFamilyName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nbcdUSB", ctypes.c_uint),
        ("nDeviceAddress", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 2),
    ]


class MV_CamL_DEV_INFO(ctypes.Structure):
    _fields_ = [
        ("chPortID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chFamilyName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", ctypes.c_uint * 38),
    ]


class MV_CML_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("chInterfaceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", ctypes.c_uint * 7),
    ]


class MV_CXP_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("chInterfaceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", ctypes.c_uint * 7),
    ]


class MV_XOF_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("chInterfaceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", ctypes.c_uint * 7),
    ]


class MV_GENTL_VIR_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("chInterfaceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chTLType", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", ctypes.c_uint * 7),
    ]


class _MV_CC_DEVICE_INFO_SPECIAL(ctypes.Union):
    _fields_ = [
        ("stGigEInfo", MV_GIGE_DEVICE_INFO),
        ("stUsb3VInfo", MV_USB3_DEVICE_INFO),
        ("stCamLInfo", MV_CamL_DEV_INFO),
        ("stCMLInfo", MV_CML_DEVICE_INFO),
        ("stCXPInfo", MV_CXP_DEVICE_INFO),
        ("stXoFInfo", MV_XOF_DEVICE_INFO),
        ("stVirInfo", MV_GENTL_VIR_DEVICE_INFO),
    ]


class MV_CC_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("nMajorVer", ctypes.c_ushort),
        ("nMinorVer", ctypes.c_ushort),
        ("nMacAddrHigh", ctypes.c_uint),
        ("nMacAddrLow", ctypes.c_uint),
        ("nTLayerType", ctypes.c_uint),
        ("nDevTypeInfo", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 3),
        ("SpecialInfo", _MV_CC_DEVICE_INFO_SPECIAL),
    ]


class MV_CC_DEVICE_INFO_LIST(ctypes.Structure):
    _fields_ = [
        ("nDeviceNum", ctypes.c_uint),
        ("pDeviceInfo", ctypes.POINTER(MV_CC_DEVICE_INFO) * MV_MAX_DEVICE_NUM),
    ]


class MV_CHUNK_DATA_CONTENT(ctypes.Structure):
    _fields_ = [
        ("pChunkData", ctypes.POINTER(ctypes.c_ubyte)),
        ("nChunkID", ctypes.c_uint),
        ("nChunkLen", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 8),
    ]


class MV_CC_IMAGE(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_uint),
        ("nHeight", ctypes.c_uint),
        ("enPixelType", MvGvspPixelType),
        ("pImageBuf", ctypes.POINTER(ctypes.c_ubyte)),
        ("nImageBufSize", ctypes.c_uint64),
        ("nImageLen", ctypes.c_uint64),
        ("nReserved", ctypes.c_uint * 4),
    ]


class _MV_FRAME_OUT_INFO_EX_UNPARSED(ctypes.Union):
    _fields_ = [
        ("pUnparsedChunkContent", ctypes.POINTER(MV_CHUNK_DATA_CONTENT)),
        ("nAligning", ctypes.c_int64),
    ]


class _MV_FRAME_OUT_INFO_EX_SUBIMAGE(ctypes.Union):
    _fields_ = [
        ("pstSubImage", ctypes.POINTER(MV_CC_IMAGE)),
        ("nAligning", ctypes.c_int64),
    ]


class _MV_FRAME_OUT_INFO_EX_USERPTR(ctypes.Union):
    _fields_ = [
        ("pUser", ctypes.c_void_p),
        ("nAligning", ctypes.c_int64),
    ]


class MV_FRAME_OUT_INFO_EX(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enPixelType", MvGvspPixelType),
        ("nFrameNum", ctypes.c_uint),
        ("nDevTimeStampHigh", ctypes.c_uint),
        ("nDevTimeStampLow", ctypes.c_uint),
        ("nReserved0", ctypes.c_uint),
        ("nHostTimeStamp", ctypes.c_int64),
        ("nFrameLen", ctypes.c_uint),
        ("nSecondCount", ctypes.c_uint),
        ("nCycleCount", ctypes.c_uint),
        ("nCycleOffset", ctypes.c_uint),
        ("fGain", ctypes.c_float),
        ("fExposureTime", ctypes.c_float),
        ("nAverageBrightness", ctypes.c_uint),
        ("nRed", ctypes.c_uint),
        ("nGreen", ctypes.c_uint),
        ("nBlue", ctypes.c_uint),
        ("nFrameCounter", ctypes.c_uint),
        ("nTriggerIndex", ctypes.c_uint),
        ("nInput", ctypes.c_uint),
        ("nOutput", ctypes.c_uint),
        ("nOffsetX", ctypes.c_ushort),
        ("nOffsetY", ctypes.c_ushort),
        ("nChunkWidth", ctypes.c_ushort),
        ("nChunkHeight", ctypes.c_ushort),
        ("nLostPacket", ctypes.c_uint),
        ("nUnparsedChunkNum", ctypes.c_uint),
        ("UnparsedChunkList", _MV_FRAME_OUT_INFO_EX_UNPARSED),
        ("nExtendWidth", ctypes.c_uint),
        ("nExtendHeight", ctypes.c_uint),
        ("nFrameLenEx", ctypes.c_uint64),
        ("nReserved1", ctypes.c_uint),
        ("nSubImageNum", ctypes.c_uint),
        ("SubImageList", _MV_FRAME_OUT_INFO_EX_SUBIMAGE),
        ("UserPtr", _MV_FRAME_OUT_INFO_EX_USERPTR),
        ("nReserved", ctypes.c_uint * 26),
    ]


class MV_FRAME_OUT(ctypes.Structure):
    _fields_ = [
        ("pBufAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
        ("nRes", ctypes.c_uint * 16),
    ]


class MV_SAVE_IMAGE_TO_FILE_PARAM_EX(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_uint),
        ("nHeight", ctypes.c_uint),
        ("enPixelType", MvGvspPixelType),
        ("pData", ctypes.POINTER(ctypes.c_ubyte)),
        ("nDataLen", ctypes.c_uint),
        ("enImageType", MV_SAVE_IMAGE_TYPE),
        ("pcImagePath", ctypes.POINTER(ctypes.c_char)),
        ("nQuality", ctypes.c_uint),
        ("iMethodValue", ctypes.c_int),
        ("nReserved", ctypes.c_uint * 8),
    ]


class MV_CC_PIXEL_CONVERT_PARAM(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enSrcPixelType", MvGvspPixelType),
        ("pSrcData", ctypes.POINTER(ctypes.c_ubyte)),
        ("nSrcDataLen", ctypes.c_uint),
        ("enDstPixelType", MvGvspPixelType),
        ("pDstBuffer", ctypes.POINTER(ctypes.c_ubyte)),
        ("nDstLen", ctypes.c_uint),
        ("nDstBufferSize", ctypes.c_uint),
        ("nRes", ctypes.c_uint * 4),
    ]


def _load_sdk():
    system = platform.system()
    if system == "Windows":
        if "winmode" in ctypes.WinDLL.__init__.__code__.co_varnames:
            return ctypes.WinDLL("MvCameraControl.dll", winmode=0)
        return ctypes.WinDLL("MvCameraControl.dll")

    base = os.getenv("MVCAM_COMMON_RUNENV")
    if not base:
        raise RuntimeError("MVCAM_COMMON_RUNENV is not set")

    arch = platform.machine()
    if arch == "aarch64":
        path = os.path.join(base, "aarch64", "libMvCameraControl.so")
    elif arch == "x86_64":
        subdir = "32" if platform.architecture()[0] == "32bit" else "64"
        path = os.path.join(base, subdir, "libMvCameraControl.so")
    elif arch == "armhf":
        path = os.path.join(base, "armhf", "libMvCameraControl.so")
    elif arch == "arm-none":
        path = os.path.join(base, "arm-none", "libMvCameraControl.so")
    else:
        raise RuntimeError("machine: %s, not support." % arch)

    return ctypes.cdll.LoadLibrary(path)


_sdk = _load_sdk()


def _setup_sdk():
    funcs = {
        "MV_CC_Initialize": ((), ctypes.c_int),
        "MV_CC_Finalize": ((), ctypes.c_int),
        "MV_CC_EnumDevices": ((ctypes.c_uint, ctypes.c_void_p), ctypes.c_uint),
        "MV_CC_CreateHandle": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
        "MV_CC_DestroyHandle": ((ctypes.c_void_p,), ctypes.c_uint),
        "MV_CC_OpenDevice": (
            (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16),
            ctypes.c_uint,
        ),
        "MV_CC_CloseDevice": ((ctypes.c_void_p,), ctypes.c_uint),
        "MV_CC_StartGrabbing": ((ctypes.c_void_p,), ctypes.c_uint),
        "MV_CC_StopGrabbing": ((ctypes.c_void_p,), ctypes.c_uint),
        "MV_CC_GetImageBuffer": (
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
            ctypes.c_uint,
        ),
        "MV_CC_FreeImageBuffer": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
        "MV_CC_SetCommandValue": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
        "MV_CC_ConvertPixelType": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
        "MV_CC_SaveImageToFileEx": ((ctypes.c_void_p, ctypes.c_void_p), ctypes.c_uint),
    }
    for name, (argtypes, restype) in funcs.items():
        func = getattr(_sdk, name)
        func.argtypes = argtypes
        func.restype = restype


_setup_sdk()


class MvCamera:
    def __init__(self):
        self._handle = ctypes.c_void_p()
        self.handle = ctypes.pointer(self._handle)

    @staticmethod
    def initialize():
        return _sdk.MV_CC_Initialize()

    @staticmethod
    def finalize():
        return _sdk.MV_CC_Finalize()

    @staticmethod
    def enum_devices(tlayer_type, dev_list):
        return _sdk.MV_CC_EnumDevices(
            ctypes.c_uint(tlayer_type), ctypes.byref(dev_list)
        )

    def create_handle(self, dev_info):
        return _sdk.MV_CC_CreateHandle(
            ctypes.byref(self.handle), ctypes.byref(dev_info)
        )

    def destroy_handle(self):
        return _sdk.MV_CC_DestroyHandle(self.handle)

    def open_device(self, access_mode=MV_ACCESS_Exclusive, switchover_key=0):
        return _sdk.MV_CC_OpenDevice(self.handle, access_mode, switchover_key)

    def close_device(self):
        return _sdk.MV_CC_CloseDevice(self.handle)

    def start_grabbing(self):
        return _sdk.MV_CC_StartGrabbing(self.handle)

    def stop_grabbing(self):
        return _sdk.MV_CC_StopGrabbing(self.handle)

    def get_image_buffer(self, frame, timeout_ms):
        return _sdk.MV_CC_GetImageBuffer(self.handle, ctypes.byref(frame), timeout_ms)

    def free_image_buffer(self, frame):
        return _sdk.MV_CC_FreeImageBuffer(self.handle, ctypes.byref(frame))

    def trigger(self):
        return _sdk.MV_CC_SetCommandValue(self.handle, b"TriggerSoftware")

    def convert_pixel_type(self, param):
        return _sdk.MV_CC_ConvertPixelType(self.handle, ctypes.byref(param))

    def save_image_to_file(self, param):
        return _sdk.MV_CC_SaveImageToFileEx(self.handle, ctypes.byref(param))


def _decode_char_array(buf):
    return bytes(buf).split(b"\x00", 1)[0].decode("ascii", errors="ignore")


def _format_ip(ip_value):
    return "%d.%d.%d.%d" % (
        (ip_value >> 24) & 0xFF,
        (ip_value >> 16) & 0xFF,
        (ip_value >> 8) & 0xFF,
        ip_value & 0xFF,
    )


def _device_summary(device_info):
    if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        model = _decode_char_array(device_info.SpecialInfo.stGigEInfo.chModelName)
        ip = _format_ip(device_info.SpecialInfo.stGigEInfo.nCurrentIp)
        return "GigE %s (%s)" % (model, ip)
    if device_info.nTLayerType == MV_USB_DEVICE:
        model = _decode_char_array(device_info.SpecialInfo.stUsb3VInfo.chModelName)
        serial = _decode_char_array(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        return "USB %s (%s)" % (model, serial)
    if device_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
        model = _decode_char_array(device_info.SpecialInfo.stCMLInfo.chModelName)
        serial = _decode_char_array(device_info.SpecialInfo.stCMLInfo.chSerialNumber)
        return "CML %s (%s)" % (model, serial)
    if device_info.nTLayerType == MV_GENTL_CXP_DEVICE:
        model = _decode_char_array(device_info.SpecialInfo.stCXPInfo.chModelName)
        serial = _decode_char_array(device_info.SpecialInfo.stCXPInfo.chSerialNumber)
        return "CXP %s (%s)" % (model, serial)
    if device_info.nTLayerType == MV_GENTL_XOF_DEVICE:
        model = _decode_char_array(device_info.SpecialInfo.stXoFInfo.chModelName)
        serial = _decode_char_array(device_info.SpecialInfo.stXoFInfo.chSerialNumber)
        return "XoF %s (%s)" % (model, serial)
    return "TLayer 0x%x" % device_info.nTLayerType


def _format_err(stage: str, ret: int) -> str:
    return f"{stage}(0x{int(ret):08X})"


def _output_pixel_type(cfg: CameraConfig) -> int:
    return (
        MV_GVSP_PIX_MONO8
        if str(cfg.output_pixel_format).lower() == "mono8"
        else MV_GVSP_PIX_BGR8_PACKED
    )


def _output_stride(cfg: CameraConfig) -> int:
    return 1 if str(cfg.output_pixel_format).lower() == "mono8" else 3


def _save_image(
    cam: MvCamera, path: str, dst_type: int, buf, buf_len: int, width: int, height: int
) -> int:
    save_param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
    save_param.enPixelType = dst_type
    save_param.nWidth = width
    save_param.nHeight = height
    save_param.nDataLen = buf_len
    save_param.pData = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    save_param.enImageType = _image_type_from_path(path)
    save_param.pcImagePath = ctypes.create_string_buffer(path.encode("ascii"))
    save_param.nQuality = 80
    save_param.iMethodValue = 1
    return cam.save_image_to_file(save_param)


def _image_type_from_path(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return MV_Image_Jpeg
    if ext == ".png":
        return MV_Image_Png
    return MV_Image_Bmp


def _format_filename(frame_id, ext, ts_utc: Optional[datetime] = None):
    ref = ts_utc or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    ref = ref.astimezone(timezone.utc)
    ts = ref.strftime("%H-%M-%S.%f")[:-3] + "Z"
    return f"{ts}_{int(frame_id):05d}{ext}"


_DATE_CACHE: dict[str, str | None] = {"date": None, "path": None}


def _convert_frame(
    cam: MvCamera, frame: MV_FRAME_OUT, cfg: CameraConfig
) -> Tuple[Optional[Tuple[object, int, int, int]], Optional[str]]:
    info = frame.stFrameInfo
    width = int(info.nWidth or info.nExtendWidth)
    height = int(info.nHeight or info.nExtendHeight)
    if width <= 0 or height <= 0:
        return None, "invalid_frame_size"
    if width > 0xFFFF or height > 0xFFFF:
        return None, "frame_size_exceeds_ushort"
    dst_type = _output_pixel_type(cfg)
    expected = width * height * _output_stride(cfg)
    buf = (ctypes.c_ubyte * expected)()
    param = MV_CC_PIXEL_CONVERT_PARAM()
    param.nWidth = width
    param.nHeight = height
    param.enSrcPixelType = info.enPixelType
    param.pSrcData = frame.pBufAddr
    param.nSrcDataLen = int(info.nFrameLenEx or info.nFrameLen)
    param.enDstPixelType = dst_type
    param.pDstBuffer = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    param.nDstBufferSize = expected
    ret = cam.convert_pixel_type(param)
    if ret != MV_OK:
        return None, _format_err("convert", ret)
    if int(param.nDstLen) > 0 and int(param.nDstLen) < expected:
        return None, "convert_size"
    return (buf, expected, width, height), None


def _process_frame(
    cam: MvCamera,
    frame: MV_FRAME_OUT,
    cfg: CameraConfig,
    frame_id: Optional[int],
    trigger_ts_utc: Optional[datetime] = None,
) -> Tuple[Optional[Tuple[Optional[str], np.ndarray, datetime]], Optional[str]]:
    converted, err = _convert_frame(cam, frame, cfg)
    if not converted:
        return None, err
    buf, buf_len, width, height = converted
    now_utc = datetime.now(timezone.utc)
    path = None
    save_ret = MV_OK
    if cfg.save_images:
        file_ts_utc = trigger_ts_utc or now_utc
        if file_ts_utc.tzinfo is None:
            file_ts_utc = file_ts_utc.replace(tzinfo=timezone.utc)
        file_ts_utc = file_ts_utc.astimezone(timezone.utc)
        date_dir = file_ts_utc.date().isoformat()
        if _DATE_CACHE["date"] != date_dir or not _DATE_CACHE["path"]:
            target_dir = os.path.join(cfg.save_dir, date_dir)
            os.makedirs(target_dir, exist_ok=True)
            _DATE_CACHE["date"] = date_dir
            _DATE_CACHE["path"] = target_dir
        target_dir = _DATE_CACHE["path"]
        file_id = frame_id if frame_id is not None else int(frame.stFrameInfo.nFrameNum)
        path = os.path.join(
            target_dir, _format_filename(file_id, cfg.ext, ts_utc=file_ts_utc)
        )
        save_ret = _save_image(
            cam, path, _output_pixel_type(cfg), buf, buf_len, width, height
        )
        if save_ret != MV_OK:
            path = None

    arr = np.ctypeslib.as_array(buf)
    if str(cfg.output_pixel_format).lower() == "mono8":
        arr = arr.reshape((height, width))
    else:
        arr = arr.reshape((height, width, 3))
    return (
        path if (cfg.save_images and save_ret == MV_OK) else None,
        arr,
        now_utc,
    ), None


@register_camera("hik")
class HikCamera(BaseCamera):
    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self.cam = MvCamera()

    def capture_once(self, idx, triggered_at: Optional[datetime] = None):
        last_err = None
        for _ in range(self.cfg.max_retry_per_frame):
            trig_ret = self.cam.trigger()
            if trig_ret != MV_OK:
                last_err = _format_err("trigger", trig_ret)
                continue

            step_t = time.perf_counter()
            frame = MV_FRAME_OUT()
            grab_ret = self.cam.get_image_buffer(frame, self.cfg.timeout_ms)
            if grab_ret != MV_OK or not frame.pBufAddr:
                last_err = _format_err("grab", grab_ret)
                continue
            grab_ms = (time.perf_counter() - step_t) * 1000

            try:
                result, err = _process_frame(
                    self.cam,
                    frame,
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
                        try:
                            rel_path = os.path.relpath(path)
                        except Exception:
                            rel_path = path
                        L.info("[%5s] @ %s", idx, rel_path)
                    else:
                        L.info("[%5s] captured (no file save)", idx)
                    return CaptureResult(
                        success=True,
                        trigger_seq=idx,
                        source="",
                        device_id=str(self.cfg.device_index),
                        path=path,
                        image=arr,
                        timings={"grab_ms": grab_ms},
                        captured_at=ts,
                    )
            finally:
                self.cam.free_image_buffer(frame)

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
        ret = MvCamera.initialize()
        if ret != MV_OK:
            raise RuntimeError(_format_err("initialize", ret))
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayer_type = (
                MV_GIGE_DEVICE
                | MV_USB_DEVICE
                | MV_GENTL_GIGE_DEVICE
                | MV_GENTL_CAMERALINK_DEVICE
                | MV_GENTL_CXP_DEVICE
                | MV_GENTL_XOF_DEVICE
            )
            ret = MvCamera.enum_devices(tlayer_type, device_list)
            if ret != MV_OK:
                raise RuntimeError(_format_err("enum_devices", ret))
            if device_list.nDeviceNum == 0:
                raise RuntimeError("No cameras found")
            if self.cfg.device_index >= device_list.nDeviceNum:
                raise RuntimeError(
                    f"Device index {self.cfg.device_index} out of range (found {device_list.nDeviceNum})"
                )

            device_info = ctypes.cast(
                device_list.pDeviceInfo[self.cfg.device_index],
                ctypes.POINTER(MV_CC_DEVICE_INFO),
            ).contents
            L.info(
                "Camera [%d] %s", self.cfg.device_index, _device_summary(device_info)
            )

            ret = self.cam.create_handle(device_info)
            if ret != MV_OK:
                raise RuntimeError(_format_err("create_handle", ret))
            try:
                ret = self.cam.open_device(MV_ACCESS_Exclusive, 0)
                if ret != MV_OK:
                    raise RuntimeError(_format_err("open_device", ret))
                ret = self.cam.start_grabbing()
                if ret != MV_OK:
                    raise RuntimeError(_format_err("start_grabbing", ret))
                yield self
            finally:
                try:
                    self.cam.stop_grabbing()
                except Exception:
                    L.warning("stop grabbing failed", exc_info=True)
                try:
                    self.cam.close_device()
                finally:
                    self.cam.destroy_handle()
        finally:
            MvCamera.finalize()


__all__ = ["HikCamera"]
