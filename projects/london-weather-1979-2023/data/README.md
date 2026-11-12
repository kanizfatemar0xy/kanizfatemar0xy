# Source data

This project uses the 16,436 daily Heathrow-area London station observations spanning 1979-01-01 to 2023-12-31, as hosted in the public repository:

https://github.com/pc1991/London-Weather-Data-From-1979-To-2023

Source file: `london_weather_data_1979_to_2023.csv`

The project notebook downloads this original CSV **when run with Internet access** and caches it here. To work offline, download the CSV from:

https://raw.githubusercontent.com/pc1991/London-Weather-Data-From-1979-To-2023/main/london_weather_data_1979_to_2023.csv

Save it next to this file, keeping the filename exactly as shown above. No substitute or synthetic dataset is included in the project distribution.

**Provenance:** European Climate Assessment & Dataset (ECA&D), station ID 1860, Heathrow, UK, as described by the public repository. Daily observational variables use ECA&D units and quality codes (0 valid; 1 suspect; 9 missing).

**Licensing caution:** ECA&D's published data policy restricts its observational station data to non-commercial research and education. We have NOT republished the full source CSV in this ZIP. Consult upstream terms and the dataset owners before redistribution or commercial use.

Attribution: Klein Tank, A. M. G. et al. (2002). Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment. *International Journal of Climatology*, 22, 1441–1453.