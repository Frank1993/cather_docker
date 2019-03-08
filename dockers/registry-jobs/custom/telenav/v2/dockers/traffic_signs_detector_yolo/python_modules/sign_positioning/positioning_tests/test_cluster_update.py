import argparse
import configparser
import os
from ftplib import FTP
import zipfile
from geopy.distance import vincenty

from first_local_runner import FirstLocalRunner
from cluster_runner import ClusterRunner
from py_orbb_metadata import orbb_definitions_pb2, orbb_localization_pb2

def read_rois(cluster_file_path):
    metadata = orbb_localization_pb2.RoiSetDB()
    with open(cluster_file_path, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata

def split_rois(roi_from_db, rois_size):
    list_of_rois = [roi_from_db[start_index:start_index+rois_size] for start_index in range(0, len(roi_from_db), rois_size)]
    return list_of_rois

def write_rois_to_file(rois, file_name):
    roi_set_db = orbb_localization_pb2.RoiSetDB()
    for roi in rois:
        copy_roi = roi_set_db.roi_from_db.add()
        copy_roi.CopyFrom(roi)
    fw = open(file_name + '.txt', 'w+')
    fb = open(file_name + '.bin', 'wb+') 
    fb.write(roi_set_db.SerializeToString())
    fw.write(str(roi_set_db))   

def generate_splited_rois(in_file, out_dir, rois_size):
    files_list = []
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    in_rois = read_rois(in_file)
    splited_rois = split_rois(in_rois.roi_from_db, rois_size)
    list_index = 0
    for rois in splited_rois:
        file_name = out_dir + "/first_" + str(list_index)
        write_rois_to_file(rois, file_name)
        list_index = list_index + 1
        files_list.append(file_name)
    return files_list

def read_cluster(cluster_file_path):
    metadata = orbb_localization_pb2.Clusters()
    with open(cluster_file_path, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata.cluster

dir = os.path.dirname(os.path.realpath(__file__))

def download_test_resources():
    directory_exists = os.path.isdir(dir + "/cluster_update_rois") and os.path.isdir(dir + "/positioning_test_rois")
    if directory_exists is False:
        ftp = FTP("10.230.2.19")
        ftp.login("orbb_ftp", "plm^2")
        ftp.cwd("/ORBB/data/test/python/")
        update_rois_archive_file_name = "cluster_update_rois.zip"
        positioning_rois_archive_file_name = "positioning_test_rois.zip"
        local_update_rois = dir + update_rois_archive_file_name
        local_positioning_rois = dir + positioning_rois_archive_file_name
        ftp.retrbinary("RETR "+ update_rois_archive_file_name, open(local_update_rois, 'wb').write)
        ftp.retrbinary("RETR "+ positioning_rois_archive_file_name, open(local_positioning_rois, 'wb').write)
        ftp.close()
        with zipfile.ZipFile(local_update_rois,"r") as zip_update_rois:
             zip_update_rois.extractall(dir)
        with zipfile.ZipFile(local_positioning_rois,"r") as zip_positioning_rois:
             zip_positioning_rois.extractall(dir)

def create_cluster_files_for_single_update_test(config_file):
    """ creates cluster files using the same rois, in two different ways, from 0 and with update"""
    config = configparser.RawConfigParser()
    config.read(config_file)
    test_dir = dir + "/cluster_update_rois/single_update_test/"

    #run first local app on all
    first_local_app_runner_all = FirstLocalRunner(
        os.path.join(dir, config.get("first_local_app_arguments_all", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_all", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_all", "first_local_app_output"))
    first_local_app_runner_all()
    #run first local app on reduced rois
    first_local_app_runner_reduced = FirstLocalRunner(
         os.path.join(dir, config.get("first_local_app_arguments_all", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_reduced", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_reduced", "first_local_app_output"))
    first_local_app_runner_reduced()
    #run first local app on updated rois
    first_local_app_runner_update = FirstLocalRunner(
         os.path.join(dir, config.get("first_local_app_arguments_all", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_update", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_update", "first_local_app_output"))
    first_local_app_runner_update()
    #run cluster app on all
    cluster_app_all = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full", "cluster_app_path")),
        test_dir + config.get("cluster_app_arguments_full", "cluster_app_input"),
        "",
        test_dir + config.get("cluster_app_arguments_full", "cluster_app_output"))
    cluster_app_all()
    #run cluster app on reduced rois
    cluster_app_reduced = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full", "cluster_app_path")),
        test_dir + config.get("first_local_app_arguments_reduced", "first_local_app_output"),
        "",
        test_dir + config.get("cluster_app_arguments_reduced", "cluster_app_output"))
    cluster_app_reduced()
    #run cluster app with update rois
    cluster_app_update = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full", "cluster_app_path")),
        test_dir + config.get("first_local_app_arguments_update", "first_local_app_output"),
        test_dir + config.get("cluster_app_arguments_reduced", "cluster_app_output"),
        test_dir + config.get("cluster_app_arguments_updated", "cluster_app_output"))
    cluster_app_update()
    clusters_full = read_cluster(test_dir + config.get("cluster_app_arguments_full", "cluster_app_output"))
    clusters_updated = read_cluster( \
        test_dir + config.get("cluster_app_arguments_updated", "cluster_app_output"))
    cluster_reduced = read_cluster( \
        test_dir + config.get("cluster_app_arguments_reduced", "cluster_app_output"))
    return clusters_full, clusters_updated, cluster_reduced


def create_cluster_files_for_multi_update_test(config_file):
    """ creates cluster files using the same rois, \
     in two different ways, from 0 and with 2 updates"""
    config = configparser.RawConfigParser()
    config.read(config_file)
    test_dir = dir + "/cluster_update_rois/multi_update_test/"

    #run first local app on all
    first_local_app_runner_all = FirstLocalRunner(
        os.path.join(dir, config.get("first_local_app_arguments_all_multi_test", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_all_multi_test", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_all_multi_test", "first_local_app_output"))
    first_local_app_runner_all()
 
    #run first local app on reduced rois
    first_local_app_runner_reduced = FirstLocalRunner(
        os.path.join(dir, config.get("first_local_app_arguments_all_multi_test", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_reduced_multi_test", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_reduced_multi_test", "first_local_app_output"))
    first_local_app_runner_reduced()
    
    #run first local app on updated rois 1
    first_local_app_runner_update1 = FirstLocalRunner(
        os.path.join(dir, config.get("first_local_app_arguments_all_multi_test", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_update1_multi_test", "first_local_app_input"),
        test_dir + config.get("first_local_app_arguments_update1_multi_test", "first_local_app_output"))     
    first_local_app_runner_update1()

    #run first local app on updated rois 2
    first_local_app_runner_update2 = FirstLocalRunner(
        os.path.join(dir, config.get("first_local_app_arguments_all_multi_test", "first_local_app_path")),
        test_dir + config.get("first_local_app_arguments_update1_multi_test", "first_local_app_output"),
        test_dir + config.get("first_local_app_arguments_update2_multi_test", "first_local_app_output"))     
    first_local_app_runner_update2()

    #run cluster app on all
    cluster_app_all = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")),
        test_dir + config.get("cluster_app_arguments_full_multi_test", "cluster_app_input"),
        "",
        test_dir + config.get("cluster_app_arguments_full_multi_test", "cluster_app_output"))
    cluster_app_all()

    #run cluster app on reduced rois
    cluster_app_reduced = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")),
        test_dir + config.get("first_local_app_arguments_reduced_multi_test", "first_local_app_output"),
        "",
        test_dir + config.get("cluster_app_arguments_reduced_multi_test", "cluster_app_output"))
    cluster_app_reduced()
    
    #run cluster app with update rois 1
    cluster_app_update1 = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")),
        test_dir + config.get("first_local_app_arguments_update1_multi_test", "first_local_app_output"),
        test_dir + config.get("cluster_app_arguments_reduced_multi_test", "cluster_app_output"),
        test_dir + config.get("cluster_app_arguments_updated1_multi_test", "cluster_app_output"))
    cluster_app_update1()

    #run cluster app with update rois 2
    cluster_app_update2 = ClusterRunner(
        os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")),
        test_dir + config.get("first_local_app_arguments_update2_multi_test", "first_local_app_output"),
        test_dir + config.get("cluster_app_arguments_updated1_multi_test", "cluster_app_output"),
        test_dir + config.get("cluster_app_arguments_updated2_multi_test", "cluster_app_output"))
    cluster_app_update2()

    clusters_full = read_cluster(test_dir + config.get("cluster_app_arguments_full_multi_test",\
     "cluster_app_output"))
    clusters_updated = read_cluster( \
        test_dir + config.get("cluster_app_arguments_updated2_multi_test", "cluster_app_output"))
    cluster_reduced = read_cluster( \
        test_dir + config.get("cluster_app_arguments_reduced_multi_test", "cluster_app_output"))
    return clusters_full, clusters_updated, cluster_reduced

def create_id_roi_ids_dict(cluster):
    cluster_dict = {}
    for cluster_element in cluster:
        cluster_dict[cluster_element.cluster_id] = cluster_element.roi_ids
    return cluster_dict

def create_list_of_used_rois(cluster):
    cluster_list = []
    for cluster_element in cluster:
        for roi_id in cluster_element.roi_ids:
            cluster_list.append(roi_id)
    return cluster_list

def compare_cluster_roi_numbers(first_cluster_set, second_cluster_set, expected_delta_number):
    """ Compares if every cluster has the same number of roi ids with an expected delta """
    matched = 0
    for first_id, first_rois in first_cluster_set.items():
        cluster_rois_from_second_set = second_cluster_set[first_id]
        if len(first_rois) == len(cluster_rois_from_second_set) + expected_delta_number:
            matched = matched + 1
    first_items_nr = len(first_cluster_set.items())
    second_items_nr = len(second_cluster_set.items())
    if first_items_nr == second_items_nr:
        print("Passed: equal number of clusters")
        print("Match results: {}/{}" .format( matched, second_items_nr))
    else: 
        print((" Failed not equal number of clusters {} vs {}")\
            .format(first_items_nr, second_items_nr))

def single_update_test(config_file):
    """ Single update test """
    print("Single update test run: ")
    clusters_full, clusters_updated, cluster_reduced = \
        create_cluster_files_for_single_update_test(config_file)
    full_cluster_ids = create_id_roi_ids_dict(clusters_full)
    updated_cluster_ids = create_id_roi_ids_dict(clusters_updated)
    reduced_cluster_ids = create_id_roi_ids_dict(cluster_reduced)
    compare_cluster_roi_numbers(updated_cluster_ids, reduced_cluster_ids, 1)

def multi_update_test(config_file):
    """ Single update test """
    print("Single update test run: ")
    clusters_full, clusters_updated, cluster_reduced = \
        create_cluster_files_for_multi_update_test(config_file)
    full_cluster_ids = create_id_roi_ids_dict(clusters_full)
    updated_cluster_ids = create_id_roi_ids_dict(clusters_updated)
    reduced_cluster_ids = create_id_roi_ids_dict(cluster_reduced)
    compare_cluster_roi_numbers(updated_cluster_ids, reduced_cluster_ids, 2)

def select_unused_rois(rois_file, cluster_file):
    unused_rois = [] 
    rois = read_rois(rois_file)
    cluster = read_cluster(cluster_file)
    cluster_rois_list = create_list_of_used_rois(cluster)
    for roi in rois.roi_from_db:
        if roi.roi_id not in cluster_rois_list:
            unused_rois.append(roi)
    return unused_rois    

def generate_merged_rois(rois_in_file_name, rois_list, rois_out_file_name):
    existing_rois = read_rois(rois_in_file_name)
    for roi in rois_list:
        copy_roi = existing_rois.roi_from_db.add()
        copy_roi.CopyFrom(roi)
    write_rois_to_file(existing_rois.roi_from_db, rois_out_file_name)

def select_all_unused_rois(start_file_index, end_file_index, file_list, rois_dir, cluster_file):
    unused_roi_list = []
    for index in range (start_file_index, end_file_index):
        current_unused_list = select_unused_rois(file_list[index]+".bin", cluster_file)
        for roi_id in current_unused_list:
            unused_roi_list.append(roi_id)
    return unused_roi_list

def iterative_cluster_update(full_rois_file_name, roi_split_size, cluster_file, rois_dir, config ):
    splited_rois_file_list = generate_splited_rois(full_rois_file_name, rois_dir, roi_split_size)
    for index, file_name in enumerate(splited_rois_file_list):
        if 0 == index:
           cluster_app = ClusterRunner(os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")),\
                                        file_name+".bin",\
                                        "",\
                                        cluster_file)
           cluster_app()
        else:
            unused_rois_list = select_all_unused_rois(0, index, splited_rois_file_list, rois_dir, cluster_file)
            generate_merged_rois(file_name+".bin", unused_rois_list, rois_dir + "merged_rois_"+str(index))
            cluster_app = ClusterRunner(os.path.join(dir, config.get("cluster_app_arguments_full_multi_test", "cluster_app_path")), \
                                        rois_dir + "merged_rois_"+str(index)+".bin",\
                                        cluster_file,\
                                        cluster_file)
            cluster_app() 


def compare_clusters(test_name, reference_cluster, actual_cluster):
    print("Run {}".format(test_name))
    matches = 0
    for reference_cluster_element in reference_cluster:
        found = False
        for actual_cluster_element in actual_cluster:
            if actual_cluster_element.type == reference_cluster_element.type:
                reference_point = (reference_cluster_element.location.latitude, reference_cluster_element.location.longitude)
                actual_point = (actual_cluster_element.location.latitude, actual_cluster_element.location.longitude)
                distance = vincenty(reference_point, actual_point).meters
                if distance <= 25:
                    found = True
                    continue
        if found == True:
            matches = matches + 1
        else: 
            print("Not found cluster {}".format(reference_cluster_element))
    print("Percentage: {}/{}".format(matches, len(reference_cluster)))

def test_case_iterative_update(rois_file, roi_size, reference_cluster, rois_dir, config):
    test_case_name = "Iterative update test " + str(roi_size)
    cluster_out = rois_dir + "cluster_out_"+str(roi_size)+".bin"
    iterative_cluster_update(rois_file, roi_size, cluster_out, rois_dir, config)
    cluster_act = read_cluster(cluster_out)
    compare_clusters(test_case_name, reference_cluster, cluster_act)          

def iterative_cluster_tests(config_file):
    config = configparser.RawConfigParser()
    config.read(config_file)
    rois_file = dir + "/cluster_update_rois/iterative_update_test/first_all_rois.bin"
    rois_dir  = dir + "/cluster_update_rois/iterative_update_test/" 
    rois_set_db = read_rois(rois_file)
    all_rois_count = len(rois_set_db.roi_from_db)
    iterative_cluster_update(rois_file, all_rois_count, rois_dir + "cluster_out_all.bin", rois_dir, config)
    cluster_ref = read_cluster(rois_dir + "cluster_out_all.bin")
    test_case_iterative_update(rois_file, 500, cluster_ref, rois_dir, config)
    test_case_iterative_update(rois_file, 1000, cluster_ref, rois_dir, config)
    test_case_iterative_update(rois_file, 2000, cluster_ref, rois_dir, config)
    test_case_iterative_update(rois_file, 3000, cluster_ref, rois_dir, config)
    test_case_iterative_update(rois_file, 4000, cluster_ref, rois_dir, config)
    
def main():
    """ config file is automation/trip_downloader/conf/test_cluster_update.cfg"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True)
    args = parser.parse_args()
    config_file = args.config_file
    download_test_resources()
    single_update_test(config_file)
    multi_update_test(config_file)
    iterative_cluster_tests(config_file)

if __name__ == "__main__":
    main()
