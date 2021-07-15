"""Utilities for interacting with FTP servers."""

import getpass
import os
import re
import shutil
import time
from datetime import datetime
from ftplib import FTP, error_perm, error_temp

import h5py
from pyhdf.SD import SD, SDC


def datetime_to_subpath(dt: datetime) -> str:
    """Get the year/day folder ICARE subpath from a datetime."""
    return os.path.join(str(dt.year), f"{dt.year}_{dt.month:02d}_{dt.day:02d}") + "/"


class ICARESession:
    """ICARESession manages a session with the ICARE FTP server. Uses a cache to minimize requests.

    To use this, you need a login with ICARE. See: https://www.icare.univ-lille.fr/data-access/data-archive-access
    """

    SUBDIR_LOOKUP = {
        "SYNC": "CALIOP/CALTRACK-5km_PAR-RB2/",  # directory of CALTRACK / PARASOL merge, has time sync data
        "CLDCLASS": "CALIOP/CALTRACK-5km_CS-2B-CLDCLASS/",  # directory of CLDCLASS level 2B dataset
        "PAR": "PARASOL/L1_B-HDF/",  # directory of PARASOL level 1B dataset
    }

    def __init__(self, temp_dir: str, max_temp_files: int = 100) -> None:
        """Create an ICARESession.

        Args:
            temp_dir: Path to the temporary directory to use as a cache.
        """
        self.temp_dir = temp_dir
        self.max_temp_files = max_temp_files
        self.temp_files = []
        self.login()
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.dir_tree = {}  # keep track of the directory tree of ICARE to cut down on FTP calls

    def __del__(self):
        self.cleanup()

    def _get_rec(self, subdict, key_list):
        """Recursive get."""
        if key_list[0] not in subdict:
            return {}
        if len(key_list) == 1:
            return subdict[key_list[0]]
        return self._get_rec(subdict[key_list[0]], key_list[1:])

    def _set_rec(self, subdict, key_list, val):
        """Recursive set."""
        if len(key_list) == 1:
            subdict[key_list[0]] = val
            return subdict
        if key_list[0] not in subdict:
            subdict[key_list[0]] = {}
        subdict[key_list[0]] = self._set_rec(subdict[key_list[0]], key_list[1:], val)
        return subdict

    def login(self) -> None:
        """Log in to the ICARE FTP server, prompting user for credentials."""
        self.ftp = FTP("ftp.icare.univ-lille1.fr")
        logged_in = False
        while not logged_in:
            try:
                if os.path.exists("icare_credentials.txt"):
                    username, password = open("icare_credentials.txt").read().split("\n")
                else:
                    username = input("ICARE Username:")
                    password = getpass.getpass("ICARE Password:")
                self.ftp.login(username, password)
                logged_in = True
                self.ftp.cwd("SPACEBORNE")
            except error_perm as e:
                print(e)

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # while os.path.exists(self.temp_dir):
        #     time.sleep(0.1)

    def listdir(self, dir_path: str) -> list:
        """List the contents of a directory, with a cache.

        Args:
            dir_path: Path to the directory

        Returns:
            listing: Directory listing
        """
        split_path = [s for s in dir_path.split("/") if s != ""]
        listing = self._get_rec(self.dir_tree, split_path)
        if listing == {}:
            try:
                nlst = self.ftp.nlst(dir_path)
            except error_temp as e:
                err_code = str(e)[:3]
                if err_code in ["421", "430", "434"]:
                    self.login()
                    nlst = self.ftp.nlst(dir_path)
                else:
                    raise FileNotFoundError(f"Could not find {dir_path} in ICARE server.")
            listing = sorted([f.split("/")[-1] for f in nlst])
            listing_dict = {}
            for l in listing:
                # check for a file extension
                if len(l) > 5 and "." in l[-5:-1]:
                    listing_dict[l] = l
                else:
                    listing_dict[l] = {}
            self.dir_tree = self._set_rec(self.dir_tree, split_path, listing_dict)
        return listing

    def get_file(self, filepath: str) -> str:
        # if the file doesn't exist, download it
        local_path = os.path.join(self.temp_dir, filepath)
        if not os.path.exists(local_path):
            # if we already have too many temp_files, delete the first one
            if len(self.temp_files) == self.max_temp_files:
                os.remove(self.temp_files[0])
                self.temp_files = self.temp_files[1:]
            self.temp_files.append(local_path)
            # recursively make directories to this file
            os.makedirs(os.path.join(self.temp_dir, os.path.split(filepath)[0]), exist_ok=True)
            temp_file = open(local_path, "wb")
            try:
                self.ftp.retrbinary("RETR " + filepath, temp_file.write)
            except error_perm:
                temp_file.close()
                self.temp_files = self.temp_files[:-1]
                raise FileNotFoundError(f"Could not find {filepath} in ICARE server.")
            except error_temp as e:
                err_code = str(e)[:3]
                if err_code in ["421", "430", "434"]:
                    self.login()
                    self.ftp.retrbinary("RETR " + filepath, temp_file.write)
                else:
                    raise FileNotFoundError(f"Could not find {filepath} in ICARE server.")
            temp_file.close()
        return local_path
