#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};

extern "C"
bool NvDsInferParseCustomEfficientDetTAO(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomEfficientDetTAO(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.size() != 4) {
        std::cerr << "Mismatch in the number of output buffers."
                  << " Expected 4 output buffers, detected in the network: "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    NvDsInferDims inferDims_p_bboxes = outputLayersInfo[1].inferDims;
    int numElements_p_bboxes = inferDims_p_bboxes.numElements;

    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    float max_bbox = 0;
    for (int i = 0; i < numElements_p_bboxes; i++) {
        if (max_bbox < p_bboxes[i])
            max_bbox = p_bboxes[i];
    }

    if (p_keep_count[0] > 0) {
        assert(!(max_bbox < 2.0));
        for (int i = 0; i < p_keep_count[0]; i++) {
            if (p_scores[i] < threshold)
                continue;

            assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

            if (p_bboxes[4*i+2] < p_bboxes[4*i] ||
                p_bboxes[4*i+3] < p_bboxes[4*i+1])
                continue;

            NvDsInferObjectDetectionInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];

            float x1 = p_bboxes[4*i];   // left
	    float y1 = p_bboxes[4*i+1]; // top
	    float x2 = p_bboxes[4*i+2]; // right
	    float y2 = p_bboxes[4*i+3]; // bottom

	    object.left   = CLIP(x1, 0, networkInfo.width - 1);
	    object.top    = CLIP(y1, 0, networkInfo.height - 1);
	    object.width  = CLIP(x2 - x1, 0, networkInfo.width - 1);
	    object.height = CLIP(y2 - y1, 0, networkInfo.height - 1);


            objectList.push_back(object);
        }
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientDetTAO);
