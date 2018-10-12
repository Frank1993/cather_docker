\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -rm //philly/philly-prod-cy4/ipgexp/telenav/object_detection.py
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -rm //philly/rr1/ipgexp/telenav/object_detection.py
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -rm //philly/gcr/pnrsy/telenav/object_detection.py

\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -cp .\object_detection.py //philly/philly-prod-cy4/ipgexp/telenav/
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -cp .\object_detection.py //philly/rr1/ipgexp/telenav/
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -cp .\object_detection.py //philly/gcr/pnrsy/telenav/

\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -ls //philly/philly-prod-cy4/ipgexp/telenav/
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -ls //philly/rr1/ipgexp/telenav/
\\scratch2\scratch\Philly\philly-fs\windows\philly-fs.ps1 -ls //philly/gcr/pnrsy/telenav/

Read-Host -Prompt "Press Enter to exit"