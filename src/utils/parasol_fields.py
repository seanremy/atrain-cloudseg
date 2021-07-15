"""Standard lists of fields to get from PARASOL files."""

# All available fields
ALL = [
    ("Data_Directional_Fields", "I443NP"),
    ("Data_Directional_Fields", "I490P"),
    ("Data_Directional_Fields", "Q490P"),
    ("Data_Directional_Fields", "U490P"),
    ("Data_Directional_Fields", "I565NP"),
    ("Data_Directional_Fields", "I670P"),
    ("Data_Directional_Fields", "Q670P"),
    ("Data_Directional_Fields", "U670P"),
    ("Data_Directional_Fields", "I763NP"),
    ("Data_Directional_Fields", "I765NP"),
    ("Data_Directional_Fields", "I865P"),
    ("Data_Directional_Fields", "Q865P"),
    ("Data_Directional_Fields", "U865P"),
    ("Data_Directional_Fields", "I910NP"),
    ("Data_Directional_Fields", "I1020NP"),
    ("Data_Fields", "NViews"),
    ("Data_Fields", "cloud_indicator"),
    ("Data_Fields", "phis"),
    ("Geolocation_Fields", "Latitude"),
    ("Geolocation_Fields", "Longitude"),
    ("Geolocation_Fields", "land_sea_flag"),
    ("Geolocation_Fields", "surface_altitude"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_01"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_02"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_03"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_04"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_05"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_06"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_07"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_08"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_09"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_10"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_11"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_12"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_13"),
    ("Quality_Flags_Directional_Fields", "Quality_Flags_14"),
]

# The default set, which contains all of the wavelengths, but leaves out quality flags and phi
DEFAULT = [
    ("Data_Directional_Fields", "I443NP"),
    ("Data_Directional_Fields", "I490P"),
    ("Data_Directional_Fields", "Q490P"),
    ("Data_Directional_Fields", "U490P"),
    ("Data_Directional_Fields", "I565NP"),
    ("Data_Directional_Fields", "I670P"),
    ("Data_Directional_Fields", "Q670P"),
    ("Data_Directional_Fields", "U670P"),
    ("Data_Directional_Fields", "I763NP"),
    ("Data_Directional_Fields", "I765NP"),
    ("Data_Directional_Fields", "I865P"),
    ("Data_Directional_Fields", "Q865P"),
    ("Data_Directional_Fields", "U865P"),
    ("Data_Directional_Fields", "I910NP"),
    ("Data_Directional_Fields", "I1020NP"),
    ("Data_Fields", "Nviews"),
    ("Data_Fields", "cloud_indicator"),
    ("Geolocation_Fields", "Latitude"),
    ("Geolocation_Fields", "Longitude"),
    ("Geolocation_Fields", "land_sea_flag"),
    ("Geolocation_Fields", "surface_altitude"),
]

# The minimal set of fields to use for debugging
MINIMAL = [
    ("Data_Directional_Fields", "I565NP"),
    ("Data_Fields", "Nviews"),
    ("Data_Fields", "cloud_indicator"),
    ("Geolocation_Fields", "Latitude"),
    ("Geolocation_Fields", "Longitude"),
]

FIELD_DICT = {
    "all": ALL,
    "default": DEFAULT,
    "minimal": MINIMAL,
}
