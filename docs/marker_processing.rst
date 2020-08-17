Marker Processing
==================

This package involves the handling of marker data, specifically performing operations such as calculating protein
expression as well as using FlowSOM for clustering.


Feature Generation
##################

This module takes the MIBI data as well as the segmented cell data, and computes the protein expression
with various transformation and normalization techniques.

.. automodule:: marker_processing.markers_feature_gen
   :members:

FlowSOM Class
#############

This class uses FlowSOM clustering for protein expression data for the various segmented cells, and returns
a list containing the cell cluster labels for each segmented cell event.

.. autoclass:: marker_processing.flowsom_clustering.ClusteringFlowSOM
   :members: