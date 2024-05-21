

#include "image_processing.h"

//#define TEST_MODE
#define QUOTE_WORKAROUND_QUOTE

CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}

#ifdef TEST_MODE
	m_net = cv::dnn::readNetFromONNX("./MnistCNN_Gray.onnx");
#else
	m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
#endif
}

CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
}

cv::Mat *CImageProcessor::GetProcImage(uint32 i)
{
	if (2 < i)
	{
		i = 2;
	}
	return m_proc_image[i];
}

static int maxSize = 50;
static int minSize = 20;

int CImageProcessor::DoProcess(cv::Mat *image)
{
	if (!image)
		return (EINVALID_PARAMETER);

	static cv::Mat grayImage, colorImage, imgCanny, binaryImage;

#ifdef TEST_MODE
	// get test image
	*image = cv::imread("./MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);
#endif

	if (image->channels() > 1)
	{
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = image->clone();
	}
	else
	{
		grayImage = image->clone();
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	cv::Mat resultImage = colorImage.clone();

	// prepare image for contour calculation
	double threshold1 = 50;
	double threshold2 = 200;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);
	cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
	cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);

	// then do contour calculations
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (unsigned int idx = 0; idx < contours.size(); idx++)
	{
		// area
		double area = cv::contourArea(contours[idx]);

		// bounding rectangle
		cv::Rect rect = cv::boundingRect(contours[idx]);

		// center of gravity
		//  center of mass
		cv::Moments moment = cv::moments(contours[idx]);
		double cx = moment.m10 / moment.m00;
		double cy = moment.m01 / moment.m00;

		// to draw counter to index idx in image
		//cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8);

		int topLeftx = rect.x;
		int topLefty = rect.y;
		int width = rect.width;
		int height = rect.height;

		if ((minSize <= width || minSize <= height) &&
				width < maxSize && height < maxSize)
		{
			int sizeCrop = (13 * std::max(width, height)) / 10;
			int topLeftxCrop = std::max(0, topLeftx + (width - sizeCrop) / 2);
			int topLeftyCrop = std::max(0, topLefty + (height - sizeCrop) / 2);

			int widthCrop = std::min(sizeCrop, binaryImage.cols - topLeftxCrop);
			int heightCrop = std::min(sizeCrop, binaryImage.rows - topLeftyCrop);

			cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heightCrop);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

			cv::Mat mnistImage = grayImage(rect);
			mnistImage = 255 - mnistImage; // invert image

#ifdef TEST_MODE
			cv::imwrite("./mnistImage.png", mnistImage);
#endif

			double min, max;
			cv::minMaxLoc(mnistImage, &min, &max);
			cv::Size classRectSize = cv::Size(28, 28);
			cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1. / (max - min), classRectSize, cv::Scalar(min));
	
#ifdef QUOTE_WORKAROUND_QUOTE			
			cv::Mat output = cv::Mat::zeros(1, 10, CV_32F);
#else
			m_net.setInput(blob);
			cv::Mat output = m_net.forward();
#endif

			// getting the number with the highest probability
			int maxDigit = 0;
			double maxVal = output.at<float>(0, 0);
			for (int i0 = 0; i0 < output.cols; i0++)
			{
				//ZaK: rather skip output for productive mode
				//std::cout << i0 << "," << output.at<float>(0, i0) << std::endl;

				if (maxVal < output.at<float>(0, i0))
				{
					maxVal = output.at<float>(0, i0);
					maxDigit = i0;
				}
			}
			// draw the number
			std::string strVal = std::to_string(maxDigit);
			putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop, topLeftyCrop - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8,
							cv::Scalar(255, 0, 0), 2);
		}
	}

#ifdef TEST_MODE
	cv::imwrite("ResultImage.png", resultImage);
#endif

	*m_proc_image[0] = resultImage.clone();
	*m_proc_image[1] = imgCanny.clone();
	*m_proc_image[2] = binaryImage.clone();

	return (SUCCESS);
}
