# Traffic Signs Facing Classifier 

**General Info**

**1.0.0**
    * initial version for traffic signs facing classification
 
**1.0.1**
    * upgraded to protobuf version 1.0.5

**1.0.3**
    * updated the MQ preprocessing to skip reading protos that have no rois (no detections).
    * added osc_details to the read image for the classifier - in case the path is not valid, the image is requested
    from OSC.