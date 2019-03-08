
import os
import apollo_python_common.proto_api as metadata
import unittests.utils.roi_ssd.resources as resources

if __name__ == "__main__":
    res_dir = resources.ensure_test_res_for_roi_ssd()
    meta_file_name = os.path.join(res_dir, 'rois.bin')
    all_types_counts, all_types_names = metadata.check_imageset(meta_file_name)
    print("all_types_counts:", all_types_counts)
    print("all_types_names:", all_types_names)
