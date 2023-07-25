#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommons.h>

// gimmes from NVIDIA's example
#include "nvdsinfer_custom_impl_Yolo/yolo.h"
#include "nvdsinfer_custom_impl_Yolo/yoloPlugins.h"

// for cv::Mat object and basic image processing
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


/* Doing my best to stay consistent the TRT programming model. Info on that (and more) see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html

There's already an example of this in /opt/nvidia/deepstream/deepstream-6.0/sources/objectDetector_Yolo, but here I integrate it into a object that's easy to use for my application.

Also, in the interest of efficiency, I add the option to construct from a prebuilt engine in a plan file.
*/


class Logger : public nvinfer1::ILogger // logger object used to create runtime & builder
{
public:
	void log(Severity severity, const char* msg)
	{
		if (severity <= nvinfer1::kWARNING)
		{
			std::cout << "[HOL' UP] ";
		}
		else
		{
			std::cout << "[INFO] ";
		}

		std::cout << msg << std::endl;
	}

} logger;

struct Detection
{
	unsigned int bbox_h;
	unsigned int bbox_w;
	unsigned int bbox_x;
	unsigned int bbox_y;
	std::string label;
	unsigned int class_confidence;
};

class Detector
{
private:
	std::string config_file;
	std::string weights_file;
	nvinfer1::ICudaEngine* infrence_engine;
	nvinfer1::IRuntime* rt; // runtime object for creating net builder for engine construction


	void createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
	{
		// creates and sets the inference engince variable
		
	}

	void loadSerializedPlan(const std::string& plan_file)
	{
		// TODO
	}


public:
	Detector(const std::string& weights, const std::string& config) // create engine
	{
		this->config_file = config;
		this->weights_file = weights;
		this->rt = nvinfer1::createInferRuntime(logger);
		
	}

	Detector(const std::string& plan_file)
	{
		// TODO
	}

	void detect(cv::Mat& input_img, std::vector<Detection>& detections)
	{
		// TODO
	}

	void putBBoxes(cv::Mat& input_img, std::vector<Detection>& detections, cv::Mat& output_img)
	{
		// TODO
	}
};


#endif
