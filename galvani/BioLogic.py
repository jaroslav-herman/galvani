# -*- coding: utf-8 -*-
"""Code to read in data files from Bio-Logic instruments"""

# SPDX-FileCopyrightText: 2013-2020 Christopher Kerr, "bcolsen"
#
# SPDX-License-Identifier: GPL-3.0-or-later

__all__ = ["MPTfileCSV", "MPTfile"]

import re
import csv
from os import SEEK_SET
import os.path
import time
from datetime import date, datetime, timedelta
from collections import defaultdict, OrderedDict
import warnings

import numpy as np

UNKNOWN_COLUMN_TYPE_HIERARCHY = ("<f8", "<f4", "<u4", "<u2", "<u1")


def fieldname_to_dtype(fieldname):
    """Converts a column header from the MPT file into a tuple of
    canonical name and appropriate numpy dtype"""

    if fieldname == "mode":
        return ("mode", np.uint8)
    elif fieldname in (
        "ox/red",
        "error",
        "control changes",
        "Ns changes",
        "counter inc.",
    ):
        return (fieldname, np.bool_)
    elif fieldname in (
        "time/s",
        "P/W",
        "(Q-Qo)/mA.h",
        "x",
        "control/V",
        "control/mA",
        "control/V/mA",
        "(Q-Qo)/C",
        "dQ/C",
        "freq/Hz",
        "|Ewe|/V",
        "|I|/A",
        "Phase(Z)/deg",
        "|Z|/Ohm",
        "Re(Z)/Ohm",
        "-Im(Z)/Ohm",
        "Re(M)",
        "Im(M)",
        "|M|",
        "Re(Permittivity)",
        "Im(Permittivity)",
        "|Permittivity|",
        "Tan(Delta)",
    ):
        return (fieldname, np.float64)
    elif fieldname in (
        "Q charge/discharge/mA.h",
        "step time/s",
        "Q charge/mA.h",
        "Q discharge/mA.h",
        "Temperature/°C",
        "Efficiency/%",
        "Capacity/mA.h",
    ):
        return (fieldname, np.float64)
    elif fieldname in ("cycle number", "I Range", "Ns", "half cycle", "z cycle"):
        return (fieldname, np.int_)
    elif fieldname in ("dq/mA.h", "dQ/mA.h"):
        return ("dQ/mA.h", np.float64)
    elif fieldname in ("I/mA", "<I>/mA"):
        return ("I/mA", np.float64)
    elif fieldname in ("Ewe/V", "<Ewe>/V", "Ecell/V", "<Ewe/V>"):
        return ("Ewe/V", np.float64)
    elif fieldname.endswith(
        (
            "/s",
            "/Hz",
            "/deg",
            "/W",
            "/mW",
            "/W.h",
            "/mW.h",
            "/A",
            "/mA",
            "/A.h",
            "/mA.h",
            "/V",
            "/mV",
            "/F",
            "/mF",
            "/uF",
            "/µF",
            "/nF",
            "/C",
            "/Ohm",
            "/Ohm-1",
            "/Ohm.cm",
            "/mS/cm",
            "/%",
        )
    ):
        return (fieldname, np.float64)
    else:
        raise ValueError("Invalid column header: %s" % fieldname)


def comma_converter(float_text):
    """Convert text to float whether the decimal point is '.' or ','"""
    trans_table = bytes.maketrans(b",", b".")
    return float(float_text.translate(trans_table))


def MPTfile(file_or_path, encoding="ascii"):
    """Opens .mpt files as numpy record arrays

    Checks for the correct headings, skips any comments and returns a
    numpy record array object and a list of comments
    """

    if isinstance(file_or_path, str):
        mpt_file = open(file_or_path, "rb")
    else:
        mpt_file = file_or_path

    magic = next(mpt_file)
    if magic not in (b"EC-Lab ASCII FILE\r\n", b"BT-Lab ASCII FILE\r\n"):
        raise ValueError("Bad first line for EC-Lab file: '%s'" % magic)

    nb_headers_match = re.match(rb"Nb header lines : (\d+)\s*$", next(mpt_file))
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError("Too few header lines: %d" % nb_headers)

    # The 'magic number' line, the 'Nb headers' line and the column headers
    # make three lines. Every additional line is a comment line.
    comments = [next(mpt_file) for i in range(nb_headers - 3)]

    fieldnames = next(mpt_file).decode(encoding).strip().split("\t")
    record_type = np.dtype(list(map(fieldname_to_dtype, fieldnames)))

    # Must be able to parse files where commas are used for decimal points
    converter_dict = dict(((i, comma_converter) for i in range(len(fieldnames))))
    mpt_array = np.loadtxt(mpt_file, dtype=record_type, converters=converter_dict)

    return mpt_array, comments


def MPTfileCSV(file_or_path):
    """Simple function to open MPT files as csv.DictReader objects

    Checks for the correct headings, skips any comments and returns a
    csv.DictReader object and a list of comments
    """

    if isinstance(file_or_path, str):
        mpt_file = open(file_or_path, "r")
    else:
        mpt_file = file_or_path

    magic = next(mpt_file)
    if magic.rstrip() != "EC-Lab ASCII FILE":
        raise ValueError("Bad first line for EC-Lab file: '%s'" % magic)

    nb_headers_match = re.match(r"Nb header lines : (\d+)\s*$", next(mpt_file))
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError("Too few header lines: %d" % nb_headers)

    # The 'magic number' line, the 'Nb headers' line and the column headers
    # make three lines. Every additional line is a comment line.
    comments = [next(mpt_file) for i in range(nb_headers - 3)]

    mpt_csv = csv.DictReader(mpt_file, dialect="excel-tab")

    expected_fieldnames = (
        [
            "mode",
            "ox/red",
            "error",
            "control changes",
            "Ns changes",
            "counter inc.",
            "time/s",
            "control/V/mA",
            "Ewe/V",
            "dq/mA.h",
            "P/W",
            "<I>/mA",
            "(Q-Qo)/mA.h",
            "x",
        ],
        [
            "mode",
            "ox/red",
            "error",
            "control changes",
            "Ns changes",
            "counter inc.",
            "time/s",
            "control/V",
            "Ewe/V",
            "dq/mA.h",
            "<I>/mA",
            "(Q-Qo)/mA.h",
            "x",
        ],
        [
            "mode",
            "ox/red",
            "error",
            "control changes",
            "Ns changes",
            "counter inc.",
            "time/s",
            "control/V",
            "Ewe/V",
            "I/mA",
            "dQ/mA.h",
            "P/W",
        ],
        [
            "mode",
            "ox/red",
            "error",
            "control changes",
            "Ns changes",
            "counter inc.",
            "time/s",
            "control/V",
            "Ewe/V",
            "<I>/mA",
            "dQ/mA.h",
            "P/W",
        ],
    )
    if mpt_csv.fieldnames not in expected_fieldnames:
        raise ValueError("Unrecognised headers for MPT file format")

    return mpt_csv, comments


VMPmodule_hdr_v1 = np.dtype(
    [
        ("shortname", "S10"),
        ("longname", "S25"),
        ("length", "<u4"),
        ("version", "<u4"),
        ("date", "S8"),
    ]
)

VMPmodule_hdr_v2 = np.dtype(
    [
        ("shortname", "S10"),
        ("longname", "S25"),
        ("max length", "<u4"),
        ("length", "<u4"),
        ("version", "<u4"),
        ("unknown2", "<u4"),  # 10 for set, log and loop, 11 for data
        ("date", "S8"),
    ]
)

# Maps from colID to a tuple defining a numpy dtype
VMPdata_colID_dtype_map = {
    1: ("mode", "<f4"),
    2: ("ox/red", "<f4"),
    3: ("error", "<f4"),
    4: ("time/s", "<f8"),
    5: ("control/V/mA", "<f4"),
    6: ("Ewe/V", "<f4"),
    7: ("dq/mA.h", "<f8"),
    8: ("I/mA", "<f4"),
    9: ("Ece/V", "<f4"),
    10: ("Aux/V", "<f4"),
    11: ("<I>/mA", "<f4"),
    12: ("log(|<I>/A|)", "<f4"),
    13: ("(Q-Qo)/mA.h", "<f8"),
    14: ("x", "<f4"),
    16: ("Analog IN 1/V", "<f4"),
    17: ("Analog IN 2/V", "<f4"),
    18: ("Analog IN 3/V", "<f4"),
    19: ("control/V", "<f4"),
    20: ("control/mA", "<f4"),
    21: ("control changes", "<f4"),
    22: ("log(|I/A|)", "<f4"),
    23: ("dQ/mA.h", "<f8"),
    24: ("cycle number", "<u4"),
    25: ("DQ/mA.h", "<f8"),
    26: ("Rapp/Ohm", "<f4"),
    27: ("Ewe-Ece/V", "<f4"),
    28: ("control/°C", "<f4"),
    29: ("T/°C", "<f4"),
    30: ("rotation rate/rpm", "<f4"),
    31: ("Ns changes", "<f4"),
    32: ("freq/Hz", "<f4"),
    33: ("|Ewe|/V", "<f4"),
    34: ("|I|/A", "<f4"),
    35: ("Phase(Z)/deg", "<f4"),
    36: ("|Z|/Ohm", "<f4"),
    37: ("Re(Z)/Ohm", "<f4"),
    38: ("-Im(Z)/Ohm", "<f4"),
    39: ("I Range", "<u2"),
    40: ("Q charge/mA.h", "<f8"),
    41: ("Q discharge/mA.h", "<f8"),
    42: ("Q charge (mA.h/g)", "<f8"),
    43: ("Q discharge (mA.h/g)", "<f8"),
    44: ("Q anodic/C", "<f8"),
    45: ("Q cathodic/C", "<f8"),
    46: ("Q anodic (C/cm²)", "<f8"),
    47: ("Q cathodic (C/cm²)", "<f8"),
    48: ("Tech Num", "<u4"),
    49: ("End buf", "<u4"),
    50: ("E0/V", "<f4"),
    51: ("Ian min/mA", "<f4"),
    52: ("Ian max/mA", "<f4"),
    53: ("Ica min/mA", "<f4"),
    54: ("Ica max/mA", "<f4"),
    55: ("Ean end/V", "<f4"),
    56: ("Eca end/V", "<f4"),
    57: ("Ece min/V", "<f4"),
    58: ("Ece max/V", "<f4"),
    59: ("Analog IN 1/V min", "<f4"),
    60: ("Analog IN 1/V max", "<f4"),
    61: ("Analog IN 2/V min", "<f4"),
    62: ("Analog IN 2/V max", "<f4"),
    63: ("Analog IN 3/V min", "<f4"),
    64: ("Analog IN 3/V max", "<f4"),
    65: ("counter inc.", "<u4"),
    66: ("I forward/mA", "<f4"),
    67: ("I reverse/mA", "<f4"),
    68: ("I delta/µA", "<f4"),
    69: ("R/Ohm", "<f4"),
    70: ("P/W", "<f4"),
    71: ("control/°C", "<f4"),
    72: ("T/°C", "<f4"),
    73: ("rotation rate/rpm", "<f4"),
    74: ("|Energy|/W.h", "<f8"),
    75: ("Analog OUT/V", "<f4"),
    76: ("<I>/mA", "<f4"),
    77: ("<Ewe>/V", "<f4"),
    78: ("Cs-2/µF-2", "<f4"),
    79: ("E step/V", "<f4"),
    80: ("Rp/Ohm", "<f4"),
    81: ("Ecorr/V", "<f4"),
    82: ("Icorr/mA", "<f4"),
    83: ("I/A/cm2", "<f4"),
    84: ("Q/C/cm2", "<f4"),
    85: ("Aux", "<f4"),
    86: ("Unk1", "<f4"),
    87: ("Unk2", "<f4"),
    88: ("Unk3", "<f4"),
    89: ("Unk4", "<f4"),
    90: ("Unk5", "<f4"),
    91: ("Unk6", "<f4"),
    92: ("Unk7", "<f4"),
    93: ("Unk8", "<f4"),
    94: ("Unk9", "<f4"),
    95: ("Unk10", "<f4"),
    96: ("|Ece|/V", "<f4"),
    97: ("|Ice|/A", "<f4"),
    98: ("Phase(Zce)/deg", "<f4"),
    99: ("|Zce|/Ohm", "<f4"),
    100: ("Re(Zce)/Ohm", "<f4"),
    101: ("-Im(Zce)/Ohm", "<f4"),
    102: ("Cce-2/µF-2", "<f4"),
    103: ("Estack/V", "<f4"),
    104: ("Istack/A", "<f4"),
    105: ("|E21|/V", "<f4"),
    106: ("Phase(Z21)/deg", "<f4"),
    107: ("|E32|/V", "<f4"),
    108: ("Phase(Z32)/deg", "<f4"),
    109: ("|Z21|/Ohm", "<f4"),
    110: ("Re(Z21)/Ohm", "<f4"),
    111: ("-Im(Z21)/Ohm", "<f4"),
    112: ("C21-2/µF-2", "<f4"),
    113: ("|Z32|/Ohm", "<f4"),
    114: ("Re(Z32)/Ohm", "<f4"),
    115: ("-Im(Z32)/Ohm", "<f4"),
    116: ("C32-2/µF-2", "<f4"),
    117: ("E21/V", "<f4"),
    118: ("E32/V", "<f4"),
    119: ("Re(Y)/Ohm-1", "<f4"),
    120: ("Im(Y)/Ohm-1", "<f4"),
    121: ("|Y|/Ohm-1", "<f4"),
    122: ("Phase(Y)/deg", "<f4"),
    123: ("Energy charge/W.h", "<f8"),
    124: ("Energy discharge/W.h", "<f8"),
    125: ("Capacitance charge/µF", "<f4"),
    126: ("Capacitance discharge/µF", "<f4"),
    127: ("Delta(Phase(Z))/%", "<f4"),
    128: ("Delta(|Z|)/%", "<f4"),
    129: ("Delta(Re(Z))/%", "<f4"),
    130: ("Delta(-Im(Z))/%", "<f4"),
    131: ("Ns", "<u2"),
    132: ("dI/dt/mA/s", "<f4"),
    133: ("Delta(mass)/g", "<f4"),
    134: ("Custom01", "<f4"),
    135: ("Custom02", "<f4"),
    136: ("Custom03", "<f4"),
    137: ("Custom04", "<f4"),
    138: ("Custom05", "<f4"),
    139: ("Custom06", "<f4"),
    140: ("Custom07", "<f4"),
    141: ("Custom08", "<f4"),
    142: ("Custom09", "<f4"),
    143: ("Custom10", "<f4"),
    144: ("Custom11", "<f4"),
    145: ("Custom12", "<f4"),
    146: ("Custom13", "<f4"),
    147: ("Custom14", "<f4"),
    148: ("Custom15", "<f4"),
    149: ("Custom16", "<f4"),
    150: ("Custom17", "<f4"),
    151: ("Custom18", "<f4"),
    152: ("Custom19", "<f4"),
    153: ("Custom20", "<f4"),
    154: ("Delta(Phase(Y))/%", "<f4"),
    155: ("Delta(|Y|)/%", "<f4"),
    156: ("Delta(Re(Y))/%", "<f4"),
    157: ("Delta(-Im(Y))/%", "<f4"),
    158: ("t low", "<f4"),
    159: ("t high", "<f4"),
    160: ("dt/dE/s/V", "<f4"),
    161: ("Delta(Phase(Z))/deg", "<f4"),
    162: ("Delta(Phase(Y))/deg", "<f4"),
    163: ("|Estack|/V", "<f4"),
    164: ("|Istack|/ABT-Lab and EC-Lab OLE COM User manual", "<f4"),
    167: ("Phase(I)/rad", "<f4"),
    168: ("Rcmp/Ohm", "<f4"),
    169: ("Cs/µF", "<f4"),
    170: ("sin ampl/V", "<f4"),
    171: ("Conductivity/S.cm-1", "<f4"),
    172: ("Cp/µF", "<f4"),
    173: ("Cp-2/µF-2", "<f4"),
    174: ("<Ewe>/V", "<f4"),
    175: ("Efficiency/%", "<f8"),
    176: ("Cycling rate charge", "<f4"),
    177: ("Cycling rate discharge", "<f4"),
    178: ("Wavelength/nm", "<f4"),
    179: ("Fluorescence/V", "<f4"),
    180: ("Fluorescence/%", "<f4"),
    181: ("Transmittance/%", "<f4"),
    182: ("CD/mdeg", "<f4"),
    183: ("ORD/mdeg", "<f4"),
    184: ("Aniso Vv/V", "<f4"),
    185: ("Aniso Vh/V", "<f4"),
    186: ("LD Vv/V", "<f4"),
    187: ("LD Vh/V", "<f4"),
    188: ("LD/deltaA", "<f4"),
    189: ("Anisotropy/Aniso", "<f4"),
    190: ("T peltier/°C", "<f4"),
    191: ("T cuvette/°C", "<f4"),
    192: ("Absorbance/AU", "<f4"),
    193: ("HV/V", "<f4"),
    194: ("Absorbance2/AU", "<f4"),
    195: ("Absorbance/V", "<f4"),
    196: ("If/mA", "<f4"),
    197: ("Ic/mA", "<f4"),
    198: ("CA/mol.L-1", "<f4"),
    199: ("CB/mol.L-1", "<f4"),
    200: ("CC/mol.L-1", "<f4"),
    201: ("CD/mol.L-1", "<f4"),
    202: ("CE/mol.L-1", "<f4"),
    203: ("CF/mol.L-1", "<f4"),
    204: ("CG/mol.L-1", "<f4"),
    205: ("CH/mol.L-1", "<f4"),
    206: ("CI/mol.L-1", "<f4"),
    207: ("CJ/mol.L-1", "<f4"),
    208: ("shot number", "<u4"),
    209: ("pad number", "<u4"),
    210: ("electrode number", "<u4"),
    211: ("E1/V", "<f4"),
    212: ("E2/V", "<f4"),
    213: ("E3/V", "<f4"),
    214: ("E4/V", "<f4"),
    215: ("E5/V", "<f4"),
    216: ("E6/V", "<f4"),
    217: ("E7/V", "<f4"),
    218: ("E8/V", "<f4"),
    219: ("E9/V", "<f4"),
    220: ("E10/V", "<f4"),
    221: ("E11/V", "<f4"),
    222: ("E12/V", "<f4"),
    223: ("E13/V", "<f4"),
    224: ("E14/V", "<f4"),
    225: ("E15/V", "<f4"),
    226: ("E16/V", "<f4"),
    227: ("E17/V", "<f4"),
    228: ("E18/V", "<f4"),
    229: ("E19/V", "<f4"),
    230: ("E20/V", "<f4"),
    231: ("E21/V", "<f4"),
    232: ("E22/V", "<f4"),
    233: ("E23/V", "<f4"),
    234: ("E24/V", "<f4"),
    235: ("E25/V", "<f4"),
    236: ("E26/V", "<f4"),
    237: ("E27/V", "<f4"),
    238: ("E28/V", "<f4"),
    239: ("E29/V", "<f4"),
    240: ("E30/V", "<f4"),
    241: ("|E1|/V", "<f4"),
    242: ("|E2|/V", "<f4"),
    243: ("|E3|/V", "<f4"),
    244: ("|E4|/V", "<f4"),
    245: ("|E5|/V", "<f4"),
    246: ("|E6|/V", "<f4"),
    247: ("|E7|/V", "<f4"),
    248: ("|E8|/V", "<f4"),
    249: ("|E9|/V", "<f4"),
    250: ("|E10|/V", "<f4"),
    251: ("|E11|/V", "<f4"),
    252: ("|E12|/V", "<f4"),
    253: ("|E13|/V", "<f4"),
    254: ("|E14|/V", "<f4"),
    255: ("|E15|/V", "<f4"),
    256: ("|E16|/V", "<f4"),
    257: ("|E17|/V", "<f4"),
    258: ("|E18|/V", "<f4"),
    259: ("|E19|/V", "<f4"),
    260: ("|E20|/V", "<f4"),
    261: ("|E21|/V", "<f4"),
    262: ("|E22|/V", "<f4"),
    263: ("|E23|/V", "<f4"),
    264: ("|E24|/V", "<f4"),
    265: ("|E25|/V", "<f4"),
    266: ("|E26|/V", "<f4"),
    267: ("|E27|/V", "<f4"),
    268: ("|E28|/V", "<f4"),
    269: ("|E29|/V", "<f4"),
    270: ("|E30|/V", "<f4"),
    271: ("Phase(Z1)/deg", "<f4"),
    272: ("Phase(Z2)/deg", "<f4"),
    273: ("Phase(Z3)/deg", "<f4"),
    274: ("Phase(Z4)/deg", "<f4"),
    275: ("Phase(Z5)/deg", "<f4"),
    276: ("Phase(Z6)/deg", "<f4"),
    277: ("Phase(Z7)/deg", "<f4"),
    278: ("Phase(Z8)/deg", "<f4"),
    279: ("Phase(Z9)/deg", "<f4"),
    280: ("Phase(Z10)/deg", "<f4"),
    281: ("Phase(Z11)/deg", "<f4"),
    282: ("Phase(Z12)/degBT", "<f4"),
    283: ("Phase(Z13)/deg", "<f4"),
    284: ("Phase(Z14)/deg", "<f4"),
    285: ("Phase(Z15)/deg", "<f4"),
    286: ("Phase(Z16)/deg", "<f4"),
    287: ("Phase(Z17)/deg", "<f4"),
    288: ("Phase(Z18)/deg", "<f4"),
    289: ("Phase(Z19)/deg", "<f4"),
    290: ("Phase(Z20)/deg", "<f4"),
    291: ("Phase(Z21)/deg", "<f4"),
    292: ("Phase(Z22)/deg", "<f4"),
    293: ("Phase(Z23)/deg", "<f4"),
    294: ("Phase(Z24)/deg", "<f4"),
    295: ("Phase(Z25)/deg", "<f4"),
    296: ("Phase(Z26)/deg", "<f4"),
    297: ("Phase(Z27)/deg", "<f4"),
    298: ("Phase(Z28)/deg", "<f4"),
    299: ("Phase(Z29)/deg", "<f4"),
    300: ("Phase(Z30)/deg", "<f4"),
    301: ("|Z1|/Ohm", "<f4"),
    302: ("|Z2|/Ohm", "<f4"),
    303: ("|Z3|/Ohm", "<f4"),
    304: ("|Z4|/Ohm", "<f4"),
    305: ("|Z5|/Ohm", "<f4"),
    306: ("|Z6|/Ohm", "<f4"),
    307: ("|Z7|/Ohm", "<f4"),
    308: ("|Z8|/Ohm", "<f4"),
    309: ("|Z9|/Ohm", "<f4"),
    310: ("|Z10|/Ohm", "<f4"),
    311: ("|Z11|/Ohm", "<f4"),
    312: ("|Z12|/Ohm", "<f4"),
    313: ("|Z13|/Ohm", "<f4"),
    314: ("|Z14|/Ohm", "<f4"),
    315: ("|Z15|/Ohm", "<f4"),
    316: ("|Z16|/Ohm", "<f4"),
    317: ("|Z17|/Ohm", "<f4"),
    318: ("|Z18|/Ohm", "<f4"),
    319: ("|Z19|/Ohm", "<f4"),
    320: ("|Z20|/Ohm", "<f4"),
    321: ("|Z21|/Ohm", "<f4"),
    322: ("|Z22|/Ohm", "<f4"),
    323: ("|Z23|/Ohm", "<f4"),
    324: ("|Z24|/Ohm", "<f4"),
    325: ("|Z25|/Ohm", "<f4"),
    326: ("|Z26|/Ohm", "<f4"),
    327: ("|Z27|/Ohm", "<f4"),
    328: ("|Z28|/Ohm", "<f4"),
    329: ("|Z29|/Ohm", "<f4"),
    330: ("|Z30|/Ohm", "<f4"),
    331: ("Re(Z1)/Ohm", "<f4"),
    332: ("Re(Z2)/Ohm", "<f4"),
    333: ("Re(Z3)/Ohm", "<f4"),
    334: ("Re(Z4)/Ohm", "<f4"),
    335: ("Re(Z5)/Ohm", "<f4"),
    336: ("Re(Z6)/Ohm", "<f4"),
    337: ("Re(Z7)/Ohm", "<f4"),
    338: ("Re(Z8)/Ohm", "<f4"),
    339: ("Re(Z9)/Ohm", "<f4"),
    340: ("Re(Z10)/Ohm", "<f4"),
    341: ("Re(Z11)/Ohm", "<f4"),
    342: ("Re(Z12)/Ohm", "<f4"),
    343: ("Re(Z13)/Ohm", "<f4"),
    344: ("Re(Z14)/Ohm", "<f4"),
    345: ("Re(Z15)/Ohm", "<f4"),
    346: ("Re(Z16)/Ohm", "<f4"),
    347: ("Re(Z17)/Ohm", "<f4"),
    348: ("Re(Z18)/Ohm", "<f4"),
    349: ("Re(Z19)/Ohm", "<f4"),
    350: ("Re(Z20)/Ohm", "<f4"),
    351: ("Re(Z21)/Ohm", "<f4"),
    352: ("Re(Z22)/Ohm", "<f4"),
    353: ("Re(Z23)/Ohm", "<f4"),
    354: ("Re(Z24)/Ohm", "<f4"),
    355: ("Re(Z25)/Ohm", "<f4"),
    356: ("Re(Z26)/Ohm", "<f4"),
    357: ("Re(Z27)/Ohm", "<f4"),
    358: ("Re(Z28)/Ohm", "<f4"),
    359: ("Re(Z29)/Ohm", "<f4"),
    360: ("Re(Z30)/Ohm", "<f4"),
    361: ("-Im(Z1)/Ohm", "<f4"),
    362: ("-Im(Z2)/Ohm", "<f4"),
    363: ("-Im(Z3)/Ohm", "<f4"),
    364: ("-Im(Z4)/Ohm", "<f4"),
    365: ("-Im(Z5)/Ohm", "<f4"),
    366: ("-Im(Z6)/Ohm", "<f4"),
    367: ("-Im(Z7)/Ohm", "<f4"),
    368: ("-Im(Z8)/Ohm", "<f4"),
    369: ("-Im(Z9)/Ohm", "<f4"),
    370: ("-Im(Z10)/Ohm", "<f4"),
    371: ("-Im(Z11)/Ohm", "<f4"),
    372: ("-Im(Z12)/Ohm", "<f4"),
    373: ("-Im(Z13)/Ohm", "<f4"),
    374: ("-Im(Z14)/Ohm", "<f4"),
    375: ("-Im(Z15)/Ohm", "<f4"),
    376: ("-Im(Z16)/Ohm", "<f4"),
    377: ("-Im(Z17)/Ohm", "<f4"),
    378: ("-Im(Z18)/Ohm", "<f4"),
    379: ("-Im(Z19)/Ohm", "<f4"),
    380: ("-Im(Z20)/Ohm", "<f4"),
    381: ("-Im(Z21)/Ohm", "<f4"),
    382: ("-Im(Z22)/Ohm", "<f4"),
    383: ("-Im(Z23)/Ohm", "<f4"),
    384: ("-Im(Z24)/Ohm", "<f4"),
    385: ("-Im(Z25)/Ohm", "<f4"),
    386: ("-Im(Z26)/Ohm", "<f4"),
    387: ("-Im(Z27)/Ohm", "<f4"),
    388: ("-Im(Z28)/Ohm", "<f4"),
    389: ("-Im(Z29)/Ohm", "<f4"),
    390: ("-Im(Z30)/Ohm", "<f4"),
    391: ("<E1>/V", "<f4"),
    392: ("<E2>/V", "<f4"),
    393: ("<E3>/V", "<f4"),
    394: ("<E4>/V", "<f4"),
    395: ("<E5>/V", "<f4"),
    396: ("<E6>/V", "<f4"),
    397: ("<E7>/V", "<f4"),
    398: ("<E8>/VBT-Lab and EC-Lab OLE COM User manual", "<f4"),
    399: ("<E9>/V", "<f4"),
    400: ("<E10>/V", "<f4"),
    401: ("<E11>/V", "<f4"),
    402: ("<E12>/V", "<f4"),
    403: ("<E13>/V", "<f4"),
    404: ("<E14>/V", "<f4"),
    405: ("<E15>/V", "<f4"),
    406: ("<E16>/V", "<f4"),
    407: ("<E17>/V", "<f4"),
    408: ("<E18>/V", "<f4"),
    409: ("<E19>/V", "<f4"),
    410: ("<E20>/V", "<f4"),
    411: ("<E21>/V", "<f4"),
    412: ("<E22>/V", "<f4"),
    413: ("<E23>/V", "<f4"),
    414: ("<E24>/V", "<f4"),
    415: ("<E25>/V", "<f4"),
    416: ("<E26>/V", "<f4"),
    417: ("<E27>/V", "<f4"),
    418: ("<E28>/V", "<f4"),
    419: ("<E29>/V", "<f4"),
    420: ("<E30>/V", "<f4"),
    421: ("Phase2/deg", "<f4"),
    422: ("Phase(Zstack)/deg", "<f4"),
    423: ("|Zstack|/Ohm", "<f4"),
    424: ("Re(Zstack)/Ohm", "<f4"),
    425: ("-Im(Zstack)/Ohm", "<f4"),
    426: ("<Estack>/V", "<f4"),
    427: ("<Istack>/mA", "<f4"),
    428: ("Potential/V", "<f4"),
    429: ("Potential/V", "<f4"),
    430: ("Phase(Zwe-ce)/deg", "<f4"),
    431: ("|Zwe-ce|/Ohm", "<f4"),
    432: ("Re(Zwe-ce)/Ohm", "<f4"),
    433: ("-Im(Zwe-ce)/Ohm", "<f4"),
    434: ("(Q-Qo)/C", "<f8"),
    435: ("dQ/C", "<f8"),
    436: ("Ece dc/V", "<f4"),
    437: ("cycle time/s", "<f8"),
    438: ("step time/s", "<f8"),
    439: ("charge time/s", "<f8"),
    440: ("discharge time/s", "<f8"),
    441: ("<Ece>/V", "<f4"),
    442: ("d(Q-Qo)/dE/mA.h/V", "<f8"),
    443: ("Capacity/mA.h", "<f8"),
    444: ("control disk/V", "<f4"),
    445: ("control disk/mA", "<f4"),
    446: ("Edisk/V", "<f4"),
    447: ("Ecedisk=Ecering/V", "<f4"),
    448: ("Idisk/mA", "<f4"),
    449: ("dQdisk/C", "<f8"),
    450: ("(Q-Qo)disk/C", "<f8"),
    451: ("cycle number", "<u4"),
    452: ("control ring/V", "<f4"),
    453: ("Ering/V", "<f4"),
    454: ("Iring/mA", "<f4"),
    455: ("(Q-Qo)ring/C", "<f8"),
    456: ("Pdisk/W", "<f4"),
    457: ("Pring/W", "<f4"),
    458: ("Edisk-Ece/V", "<f4"),
    459: ("Ering-Ece/V", "<f4"),
    460: ("<time>/s", "<f8"),
    461: ("<EweX>/V", "<f4"),
    462: ("Temperature/°C", "<f4"),
    463: ("Ramp upwards", "<u1"),
    464: ("Time/µs", "<f4"),
    465: ("I Range disk", "<u2"),
    466: ("I Range ring", "<u2"),
    467: ("Q charge/discharge/mA.h", "<f8"),
    468: ("half cycle", "<u4"),
    469: ("z cycle", "<u4"),
    470: ("It/mA", "<f4"),
    471: ("<Ece>/V", "<f4"),
    472: ("Vcorr/mm/yr", "<f4"),
    473: ("THD Ewe/%", "<f4"),
    474: ("THD I/%", "<f4"),
    475: ("THD Ece/%", "<f4"),
    476: ("NSD Ewe/%", "<f4"),
    477: ("NSD I/%", "<f4"),
    478: ("NSD Ece/%", "<f4"),
    479: ("NSR Ewe/%", "<f4"),
    480: ("NSR I/%", "<f4"),
    481: ("NSR Ece/%", "<f4"),
    482: ("ShuntIsChanging", "<u1"),
    483: ("ModeIsChanging", "<u1"),
    484: ("NbIterInstr", "<u4"),
    485: ("Instr", "<f4"),
    486: ("|Ewe h2|/V", "<f4"),
    487: ("|Ewe h3|/V", "<f4"),
    488: ("|Ewe h4|/V", "<f4"),
    489: ("|Ewe h5|/V", "<f4"),
    490: ("|Ewe h6|/V", "<f4"),
    491: ("|Ewe h7|/V", "<f4"),
    492: ("|I h2|/A", "<f4"),
    493: ("|I h3|/A", "<f4"),
    494: ("|I h4|/A", "<f4"),
    495: ("|I h5|/A", "<f4"),
    496: ("|I h6|/A", "<f4"),
    497: ("|I h7|/A", "<f4"),
    498: ("|Ece h2|/V", "<f4"),
    499: ("|Ece h3|/V", "<f4"),
    500: ("|Ece h4|/V", "<f4"),
    501: ("|Ece h5|/V", "<f4"),
    502: ("|Ece h6|/V", "<f4"),
    503: ("|Ece h7|/V", "<f4"),
    504: ("Rac/Ohm", "<f4"),
    505: ("Rdc/Ohm", "<f4"),
    506: ("TCU control/°C", "<f4"),
    507: ("TCU meas. /°C", "<f4"),
    508: ("Regulation", "<u1"),
    509: ("Acir/Dcir Control", "<u1"),
    510: ("LTime/s", "<f8"),
    511: ("Re(C)/nF", "<f4"),
    512: ("Im(C)/nF", "<f4"),
    513: ("|C|/nF", "<f4"),
    514: ("Phase(C)/degBT-Lab and EC-Lab OLE COM User manual", "<f4"),
    515: ("Re(M)", "<f4"),
    516: ("Im(M)", "<f4"),
    517: ("|M|", "<f4"),
    518: ("Phase(M)/deg", "<f4"),
    519: ("Re(Permittivity)", "<f4"),
    520: ("Im(Permittivity)", "<f4"),
    521: ("|Permittivity|", "<f4"),
    522: ("Phase(Permittivity)/deg", "<f4"),
    523: ("Re(Conductivity)/mS/cm", "<f4"),
    524: ("Im(Conductivity)/mS/cm", "<f4"),
    525: ("|Conductivity|/mS/cm", "<f4"),
    526: ("Phase(Conductivity)/deg", "<f4"),
    527: ("Re(Resistivity)/Ohm.cm", "<f4"),
    528: ("Im(Resistivity)/Ohm.cm", "<f4"),
    529: ("|Resistivity|/Ohm.cm", "<f4"),
    530: ("Phase(Resistivity)/deg", "<f4"),
    531: ("Tan(Delta)", "<f4"),
    532: ("Loss Angle(Delta)/deg", "<f4"),
    533: ("TCU base /°C", "<f4"),
    534: ("TCU cell /°C", "<f4"),
    535: ("TCU sample/°C", "<f4"),
    536: ("Ewe initial/V", "<f4"),
    537: ("Ewe final/V", "<f4"),
    538: ("I initial/mA", "<f4"),
    539: ("I final/mA", "<f4"),
    540: ("P min/W", "<f4"),
    541: ("P max/W", "<f4"),
    542: ("T min/°C", "<f4"),
    543: ("T max/°C", "<f4"),
}

# These column IDs define flags which are all stored packed in a single byte
# The values in the map are (name, bitmask, dtype)
VMPdata_colID_flag_map = {
    1: ("mode", 0x03, np.uint8),
    2: ("ox/red", 0x04, np.bool_),
    3: ("error", 0x08, np.bool_),
    21: ("control changes", 0x10, np.bool_),
    31: ("Ns changes", 0x20, np.bool_),
    65: ("counter inc.", 0x80, np.bool_),
}


def parse_BioLogic_date(date_text):
    """Parse a date from one of the various formats used by Bio-Logic files."""
    date_formats = ["%m/%d/%y", "%m-%d-%y", "%m.%d.%y"]
    if isinstance(date_text, bytes):
        date_string = date_text.decode("ascii")
    else:
        date_string = date_text
    for date_format in date_formats:
        try:
            tm = time.strptime(date_string, date_format)
        except ValueError:
            continue
        else:
            break
    else:
        raise ValueError(
            f"Could not parse timestamp {date_string!r}"
            f" with any of the formats {date_formats}"
        )
    return date(tm.tm_year, tm.tm_mon, tm.tm_mday)


def VMPdata_dtype_from_colIDs(colIDs, error_on_unknown_column: bool = True):
    """Get a numpy record type from a list of column ID numbers.

    The binary layout of the data in the MPR file is described by the sequence
    of column ID numbers in the file header. This function converts that
    sequence into a list that can be used with numpy dtype load data from the
    file with np.frombuffer().

    Some column IDs refer to small values which are packed into a single byte.
    The second return value is a dict describing the bit masks with which to
    extract these columns from the flags byte.

    If error_on_unknown_column is True, an error will be raised if an unknown
    column ID is encountered. If it is False, a warning will be emited and attempts
    will be made to read the column with a few different dtypes.


    """
    type_list = []
    field_name_counts = defaultdict(int)
    flags_dict = OrderedDict()
    for colID in colIDs:
        if colID in VMPdata_colID_flag_map:
            # Some column IDs represent boolean flags or small integers
            # These are all packed into a single 'flags' byte whose position
            # in the overall record is determined by the position of the first
            # column ID of flag type. If there are several flags present,
            # there is still only one 'flags' int
            if "flags" not in field_name_counts:
                type_list.append(("flags", "u1"))
                field_name_counts["flags"] = 1
            flag_name, flag_mask, flag_type = VMPdata_colID_flag_map[colID]
            # TODO what happens if a flag colID has already been seen
            # i.e. if flag_name is already present in flags_dict?
            # Does it create a second 'flags' byte in the record?
            flags_dict[flag_name] = (np.uint8(flag_mask), flag_type)
        elif colID in VMPdata_colID_dtype_map:
            field_name, field_type = VMPdata_colID_dtype_map[colID]
            field_name_counts[field_name] += 1
            count = field_name_counts[field_name]
            if count > 1:
                unique_field_name = "%s %d" % (field_name, count)
            else:
                unique_field_name = field_name
            type_list.append((unique_field_name, field_type))
        else:
            if error_on_unknown_column:
                raise NotImplementedError(
                    "Column ID {cid} after column {prev} is unknown".format(
                        cid=colID, prev=type_list[-1][0]
                    )
                )
            warnings.warn(
                "Unknown column ID %d -- will attempt to read as common dtypes"
                % colID
            )
            type_list.append(("unknown_colID_%d" % colID, UNKNOWN_COLUMN_TYPE_HIERARCHY[0]))

    return type_list, flags_dict


def read_VMP_modules(fileobj, read_module_data=True):
    """Reads in module headers in the VMPmodule_hdr format. Yields a dict with
    the headers and offset for each module.

    N.B. the offset yielded is the offset to the start of the data i.e. after
    the end of the header. The data runs from (offset) to (offset+length)"""
    while True:
        module_magic = fileobj.read(len(b"MODULE"))
        if len(module_magic) == 0:  # end of file
            break
        elif module_magic != b"MODULE":
            raise ValueError(
                "Found %r, expecting start of new VMP MODULE" % module_magic
            )
        VMPmodule_hdr = VMPmodule_hdr_v1

        # Reading headers binary information
        hdr_bytes = fileobj.read(VMPmodule_hdr.itemsize)
        if len(hdr_bytes) < VMPmodule_hdr.itemsize:
            raise IOError("Unexpected end of file while reading module header")

        # Checking if EC-Lab version is >= 11.50
        if hdr_bytes[35:39] == b"\xff\xff\xff\xff":
            VMPmodule_hdr = VMPmodule_hdr_v2
            hdr_bytes += fileobj.read(VMPmodule_hdr_v2.itemsize - VMPmodule_hdr_v1.itemsize)

        hdr = np.frombuffer(hdr_bytes, dtype=VMPmodule_hdr, count=1)
        hdr_dict = dict(((n, hdr[n][0]) for n in VMPmodule_hdr.names))
        hdr_dict["offset"] = fileobj.tell()
        if read_module_data:
            hdr_dict["data"] = fileobj.read(hdr_dict["length"])
            if len(hdr_dict["data"]) != hdr_dict["length"]:
                raise IOError(
                    """Unexpected end of file while reading data
                    current module: %s
                    length read: %d
                    length expected: %d"""
                    % (
                        hdr_dict["longname"],
                        len(hdr_dict["data"]),
                        hdr_dict["length"],
                    )
                )
            yield hdr_dict
        else:
            yield hdr_dict
            fileobj.seek(hdr_dict["offset"] + hdr_dict["length"], SEEK_SET)


def loop_from_file(file: str, encoding: str = "latin1"):
    """
    When an experiment is still running and it includes loops,
    a _LOOP.txt file is temporarily created to progressively store the indexes of new loops.
    This function reads the file and creates the loop_index array for MPRfile initialization.

    Parameters
    ----------
    file : str
        Path of the loop file.
    encoding : str, optional
        Encoding of the text file. The default is "latin1".

    Raises
    ------
    ValueError
        If the file does not start with "VMP EXPERIMENT LOOP INDEXES".

    Returns
    -------
    loop_index : np.array
        Indexes of data points that start a new loop.

    """
    with open(file, "r", encoding=encoding) as f:
        line = f.readline().strip()
        if line != LOOP_MAGIC:
            raise ValueError("Invalid magic for LOOP.txt file")
        loop_index = np.array([int(line) for line in f], dtype="u4")

    return loop_index


def timestamp_from_file(file: str, encoding: str = "latin1"):
    """
    When an experiment is still running, a .mpl file is temporarily created to store
    information that will be added in the log module and will be appended to the data
    module in the .mpr file at the end of experiment.
    This function reads the file and extracts the experimental starting date and time
    as a timestamp for MPRfile initialization.

    Parameters
    ----------
    file : str
        Path of the log file.
    encoding : str, optional
        Encoding of the text file. The default is "latin1".

    Raises
    ------
    ValueError
        If the file does not start with "EC-Lab LOG FILE" or "BT-Lab LOG FILE".

    Returns
    -------
    timestamp
        Date and time of the start of data acquisition
    """
    with open(file, "r", encoding=encoding) as f:
        line = f.readline().strip()
        if line not in LOG_MAGIC:
            raise ValueError("Invalid magic for .mpl file")
        log = f.read()
    start = tuple(
        map(
            int,
            re.findall(
                r"Acquisition started on : (\d+)\/(\d+)\/(\d+) (\d+):(\d+):(\d+)\.(\d+)",
                "".join(log),
            )[0],
        )
    )
    return datetime(
        int(start[2]), start[0], start[1], start[3], start[4], start[5], start[6] * 1000
    )


LOG_MAGIC = "EC-Lab LOG FILEBT-Lab LOG FILE"
LOOP_MAGIC = "VMP EXPERIMENT LOOP INDEXES"
MPR_MAGIC = b"BIO-LOGIC MODULAR FILE\x1a".ljust(48) + b"\x00\x00\x00\x00"


class MPRfile:
    """Bio-Logic .mpr file

    The file format is not specified anywhere and has therefore been reverse
    engineered. Not all the fields are known.

    Attributes
    ==========
    modules - A list of dicts containing basic information about the 'modules'
              of which the file is composed.
    data - numpy record array of type VMPdata_dtype containing the main data
           array of the file.
    startdate - The date when the experiment started
    enddate - The date when the experiment finished
    """

    def __init__(self, file_or_path, error_on_unknown_column: bool = True):
        """Pass an EC-lab .mpr file to be parsed.

        Parameters:
            file_or_path: Either the open file data or a path to it.
            error_on_unknown_column: Whether or not to raise an error if an
                unknown column ID is encountered. A warning will be emited and
                the column will be added 'unknown_<colID>', with an attempt to read
                it with a few different dtypes.

        """
        self.loop_index = None
        if isinstance(file_or_path, str):
            mpr_file = open(file_or_path, "rb")
            loop_file = file_or_path[:-4] + "_LOOP.txt"  # loop file for running experiment
            log_file = file_or_path[:-1] + "l"  # log file for runnning experiment
        else:
            mpr_file = file_or_path
        magic = mpr_file.read(len(MPR_MAGIC))
        if magic != MPR_MAGIC:
            raise ValueError("Invalid magic for .mpr file: %s" % magic)

        modules = list(read_VMP_modules(mpr_file))

        self.modules = modules
        (settings_mod,) = (m for m in modules if m["shortname"] == b"VMP Set   ")
        (data_module,) = (m for m in modules if m["shortname"] == b"VMP data  ")
        maybe_loop_module = [m for m in modules if m["shortname"] == b"VMP loop  "]
        maybe_log_module = [m for m in modules if m["shortname"] == b"VMP LOG   "]

        n_data_points = np.frombuffer(data_module["data"][:4], dtype="<u4")
        n_columns = np.frombuffer(data_module["data"][4:5], dtype="u1").item()

        if data_module["version"] == 0:
            # If EC-Lab version >= 11.50, column_types is [0 1 0 3 0 174...] instead of [1 3 174...]
            if np.frombuffer(data_module["data"][5:6], dtype="u1").item():
                column_types = np.frombuffer(data_module["data"][5:], dtype="u1", count=n_columns)
                remaining_headers = data_module["data"][5 + n_columns:100]
                main_data = data_module["data"][100:]
            else:
                column_types = np.frombuffer(
                    data_module["data"][5:], dtype="u1", count=n_columns * 2
                )
                column_types = column_types[1::2]  # suppressing zeros in column types array
                # remaining headers should be empty except for bytes 5 + n_columns * 2
                # and 1006 which are sometimes == 1
                remaining_headers = data_module["data"][6 + n_columns * 2:1006]
                main_data = data_module["data"][1007:]
        elif data_module["version"] in [2, 3]:
            column_types = np.frombuffer(data_module["data"][5:], dtype="<u2", count=n_columns)
            # There are bytes of data before the main array starts
            if data_module["version"] == 3:
                num_bytes_before = 406  # version 3 added `\x01` to the start
            else:
                num_bytes_before = 405
            remaining_headers = data_module["data"][5 + 2 * n_columns:405]
            main_data = data_module["data"][num_bytes_before:]
        else:
            raise ValueError(
                "Unrecognised version for data module: %d" % data_module["version"]
            )

        assert not any(remaining_headers)

        dtypes, self.flags_dict = VMPdata_dtype_from_colIDs(
            column_types, error_on_unknown_column=error_on_unknown_column
        )

        unknown_cols = []
        # Iteratively work through the unknown columns and try to read them
        if not error_on_unknown_column:
            for col, _ in dtypes:
                if col.startswith("unknown_colID"):
                    unknown_cols.append(col)
            if len(unknown_cols) > 3:
                raise RuntimeError(
                    "Too many unknown columns to attempt to read combinatorially: %s"
                    % unknown_cols
                )

        if unknown_cols:
            # create a list of all possible combinations of dtypes
            # for the unknown columns
            from itertools import product
            perms = product(UNKNOWN_COLUMN_TYPE_HIERARCHY, repeat=len(unknown_cols))
            for perm in perms:
                for unknown_col_ind, c in enumerate(unknown_cols):
                    for ind, (col, _) in enumerate(dtypes):
                        if c == col:
                            dtypes[ind] = (col, perm[unknown_col_ind])

                try:
                    self.dtype = np.dtype(dtypes)
                    self.data = np.frombuffer(main_data, dtype=self.dtype)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(
                    "Unable to read data for unknown columns %s with any of the common dtypes %s",
                    unknown_cols,
                    UNKNOWN_COLUMN_TYPE_HIERARCHY
                )

        else:
            self.dtype = np.dtype(dtypes)
            self.data = np.frombuffer(main_data, dtype=self.dtype)

        assert self.data.shape[0] == n_data_points

        # No idea what these 'column types' mean or even if they are actually
        # column types at all
        self.version = int(data_module["version"])
        self.cols = column_types
        self.npts = n_data_points
        self.startdate = parse_BioLogic_date(settings_mod["date"])

        if maybe_loop_module:
            (loop_module,) = maybe_loop_module
            if loop_module["version"] == 0:
                self.loop_index = np.frombuffer(loop_module["data"][4:], dtype="<u4")
                self.loop_index = np.trim_zeros(self.loop_index, "b")
            else:
                raise ValueError(
                    "Unrecognised version for data module: %d" % data_module["version"]
                )
        else:
            if os.path.isfile(loop_file):
                self.loop_index = loop_from_file(loop_file)
                if self.loop_index[-1] < n_data_points:
                    self.loop_index = np.append(self.loop_index, n_data_points)

        if maybe_log_module:
            (log_module,) = maybe_log_module
            self.enddate = parse_BioLogic_date(log_module["date"])

            # There is a timestamp at either 465 or 469 bytes
            # I can't find any reason why it is one or the other in any
            # given file
            ole_timestamp1 = np.frombuffer(
                log_module["data"][465:], dtype="<f8", count=1
            )
            ole_timestamp2 = np.frombuffer(
                log_module["data"][469:], dtype="<f8", count=1
            )
            ole_timestamp3 = np.frombuffer(
                log_module["data"][473:], dtype="<f8", count=1
            )
            ole_timestamp4 = np.frombuffer(
                log_module["data"][585:], dtype="<f8", count=1
            )

            if ole_timestamp1 > 40000 and ole_timestamp1 < 50000:
                ole_timestamp = ole_timestamp1
            elif ole_timestamp2 > 40000 and ole_timestamp2 < 50000:
                ole_timestamp = ole_timestamp2
            elif ole_timestamp3 > 40000 and ole_timestamp3 < 50000:
                ole_timestamp = ole_timestamp3
            elif ole_timestamp4 > 40000 and ole_timestamp4 < 50000:
                ole_timestamp = ole_timestamp4

            else:
                raise ValueError("Could not find timestamp in the LOG module")

            ole_base = datetime(1899, 12, 30, tzinfo=None)
            ole_timedelta = timedelta(days=ole_timestamp[0])
            self.timestamp = ole_base + ole_timedelta
            if self.startdate != self.timestamp.date():
                raise ValueError(
                    "Date mismatch:\n"
                    + "    Start date: %s\n" % self.startdate
                    + "    End date: %s\n" % self.enddate
                    + "    Timestamp: %s\n" % self.timestamp
                )
        else:
            if os.path.isfile(log_file):
                self.timestamp = timestamp_from_file(log_file)
                self.enddate = None

    def get_flag(self, flagname):
        if flagname in self.flags_dict:
            mask, dtype = self.flags_dict[flagname]
            return np.array(self.data["flags"] & mask, dtype=dtype)
        else:
            raise AttributeError("Flag '%s' not present" % flagname)
