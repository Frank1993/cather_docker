maintainer: adip


**A. Setting up the config file**
    [list] arguments

    1. signs: list of comma separated signposts to be included in the download process
        e.g signs:TRAFFIC_LIGHTS_SIGN,TURN_RESTRICTION_LEFT_US,STOP_US,SIGNPOST_GENERIC
    USE 'all' to include all the signpost in the download process
        e.g signs:all

    2. regions: list of comma separated regions to be included in the download process
        currently working only with regions set to 'all'
        regions:all

    3. trip_ids_included: list of comma separated trip id's to be downloaded
        e.g trip_ids_included:928603,439654,588825,309755,440821,67144
       USE 'all' to download all the trips
        e.g trip_ids_included:all

    4. trip_ids_excluded: list of comma separated trip id's to be excluded from the download process
        e.g trip_ids_excluded:928603,439654,588825,309755,440821,67144
       USE 'all' to not exclude any trip
        e.g trip_ids_excluded:all

    [flags] yes/no

    5. manual: yes to include detections obtained from tagging, no otherwise

    6. automatic: yes to include detections made by the detector components, no otherwise

    7. confirmed: yes to include all the components which confirmation's status is marked as valid, no otherwise

    8. removed: yes to include all the components which confirmation's status is marked as invalid, no otherwise

    9. to_be_checked: yes to include all the components without a confirmation status, no otherwise

    10. full_trips: yes to download all the photos from the selected list of trips, no otherwise
        won't work with the trips_ids_included set to all
    11. sign_components: yes to include all signpost's components in the download, no otherwise

    12. telenav_user: yes to download only components which status was marked as valid/invalid by a telenav user, no otherwise

    13. proto_rois_only: yes to download only the metadata without images, no to download both images and metadata

    14. remove_duplicates: yes to remove all the duplicates of signposts inside images, no otherwise

    15. min_sign_size: (int value) filter by minimum number of pixels each edge of a roi must have, won't be included if
        at least one edge is lower than min_sign_size value
         e.g min_sign_size: 4
    16. gte_timestamp: greater than or equal timestamp for photo creation (format YYYY-MM-DD HH:MM:SS)
         e.g gte_timestamp: 2015-09-09 12:12:59
        USE bot(beginning of time) to download from day one
         e.g gte_timestamp: bot

    17. lte_timestamp: less than or equal timestamp for photo creation, (format YYYY-MM-DD HH:MM:SS)
        e.g lte_timestamp: 2015-09-09 12:12:59
        USE now to download everything until current time
        e.g lte_timestamp: now

**B.Set up download_trips.sh**
     -CONFIG_FILE set to the config file's path (config.cfg)
     -DOWNLOAD_FOLDER set to the download's file path
     -THREADS_NUMBER set between 1/maximum value according to user processor thread number
     -UPDATE_IMAGE_SET_FILE set to a .bin to be updated with current components

**C.Run the download_trips.sh**
    `chmod +x ./download_trips.sh` followed by `./download_trips.sh`