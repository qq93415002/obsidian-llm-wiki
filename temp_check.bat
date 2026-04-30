@echo off
powershell -Command "Get-WmiObject MSAcpi_ThermalZoneTemperature | ForEach-Object { Write-Host ($_.Name + ': ' + [math]::Round($_.CurrentTemperature/10-273.15,1) + 'C') }"
pause
