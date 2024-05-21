

#include "image_processing.h"
//#define TEST_MODE

CImageProcessor::CImageProcessor() {
	#ifdef TEST_MODE
		m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
	#else
		m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
	#endif

	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		delete m_proc_image[i];
	}
}

cv::Mat* CImageProcessor::GetProcImage(uint32 i) {
	if(2 < i) {
		i = 2;
	}
	return m_proc_image[i];
}

int CImageProcessor::DoProcess(cv::Mat* image) {
	cv::Mat grayImage;
	cv::Mat colorImage;

	if(!image) return(EINVALID_PARAMETER);	
		#ifdef TEST_MODE
			*image = cv::imread("../CNNdir/MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);
		#endif
		
		if(image->channels() > 1) {
			cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
			colorImage = *image;
		}
		else {
			grayImage = *image;
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}

		double threshold1 = 50;
		double threshold2 = 200;

		cv::Mat imgCanny;
		cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

		cv::Mat binaryImage;
		cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
		cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		// connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
		cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL , cv::CHAIN_APPROX_SIMPLE);

		int minSize = 20;
		int maxSize = 50;

		// for visualisation
		cv::Mat resultImage = colorImage.clone();

		for(unsigned int idx = 0; idx < contours.size(); idx++)
		{
			cv::Rect boundingBox = cv::boundingRect(contours[idx]);
			int width = boundingBox.width;
			int height = boundingBox.height;
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;
			int topLeftx = (int) cx - width/2;
			int topLefty = (int) cy - height/2;
			/*
			cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );
			cv::Rect rect = cv::boundingRect(contours[idx]);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
			*/
			
			if ((minSize <= width || minSize <= height) && width <= maxSize && height <= maxSize)
			{
				// DEFINE BOXES WHERE THE NUMBERS ARE ON THE IMAGE
				int sizeCrop = (13*std::max(width, height))/10;
				// Top left corner for crop
				int topLeftxCrop = std::max(0, topLeftx+(width-sizeCrop)/2);
				int topLeftyCrop = std::max(0, topLefty+(height-sizeCrop)/2);
				// Ensure the whole crop lies on image
				int widthCrop = std::min(sizeCrop, binaryImage.cols-topLeftxCrop);
				int heightCrop = std::min(sizeCrop, binaryImage.rows-topLeftyCrop);
				// Crop Box
				cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heightCrop);
				// Draw Crop Box
				cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

				// FEED THE PARTIAL IMAGES INTO THE FNN
				// Crop the parts out of the image
				cv::Mat mnistImage = grayImage(rect);
				// Invert image
				mnistImage = 255 - mnistImage;
				// Scaling
				double min, max;
				cv::minMaxLoc(mnistImage, &min, &max);
				cv::Size classRectSize = cv::Size(28, 28);
				cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1./(max-min), classRectSize, cv::Scalar(min));
				// Classification
				m_net.setInput(blob);
				cv::Mat output = m_net.forward();

				int maxDigit = 0;
				float maxValue = 0.0;
				for(int i0 = 0; i0 < output.cols; i0++)
				{
					if (output.at<float>(0,i0) > maxValue)
					{
						maxValue = output.at<float>(0,i0);
						maxDigit = i0;
					}
					// std::cout << i0 << "," << output.at<float>(0,i0) << std::endl;
				}

				
				std::string strVal = std::to_string(maxDigit);
				putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop, topLeftyCrop-5),
						cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

			} 
		}
			#ifdef TEST_MODE
				cv::imwrite("resultImage.png", resultImage);
			#endif


			// Show
			*m_proc_image[0] = resultImage;
			*m_proc_image[1] = imgCanny;
			*m_proc_image[2] = binaryImage;
		
		// mPrevImage = grayImage.clone();


        // cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);
		//  cv::imwrite("dx.png", *m_proc_image[0]);
		//  cv::imwrite("dy.png", *m_proc_image[1]);

	return(SUCCESS);
}









