

#include "image_processing.h"

//#define TEST_MODE
#define RASPBERR_PI
//#define REGION_LABELING

CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}

	#ifdef RASPBERR_PI
		m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
	#else
		m_net = cv::dnn::readNetFromONNX("MnistCNN_Gray.onnx");
	#endif
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
	
	if(!image) return(EINVALID_PARAMETER);	
		

		#ifdef TEST_MODE
			#ifdef RASPBERR_PI
				*image = cv::imread("/home/pi/CNNdir/MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);
			#else
				*image = cv::imread("MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);				
			#endif
		#endif

		cv::Mat grayImage;
		cv::Mat colorImage;

		if(image->channels() > 1){
			cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
			colorImage = *image;
		}else{
			grayImage = *image;
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}

		double threshold1 = 50;
		double threshold2 = 200;
		cv::Mat imgCanny;
		cv::Mat binaryImage;
		cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

		cv::Mat kernel = cv::Mat::ones(5,5, CV_8UC1);
		cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);
		
		#ifdef REGION_LABELING
			cv::Mat stats, centroids, labelImage;
			connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

			int minSize = 20;
			int maxSize = 50;

			cv::Mat resultImage = colorImage.clone();

			for (int i = 1; i < stats.rows; i++){
				int topLeftx = stats.at<int>(i,0);
				int topLefty = stats.at<int>(i, 1);
				int width = stats.at<int>(i, 2);
				int heigth = stats.at<int>(i, 3);
				
				int area = stats.at<int>(i, 4);

				double cx = centroids.at<double>(i,0);
				double cy = centroids.at<double>(i,1);

				if((minSize <= width || minSize <= heigth) && width < maxSize && heigth < maxSize){
					int sizeCrop = (13*std::max(width, heigth))/10;
					int topLeftxCrop = std::max(0, topLeftx + (width - sizeCrop)/2);
					int topLeftyCrop = std::max(0, topLefty + (width - sizeCrop)/2);

					int widthCrop = std::min(sizeCrop, binaryImage.cols - topLeftxCrop);
					int heigthCrop = std::min(sizeCrop, binaryImage.rows - topLeftyCrop);

					cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heigthCrop);

					cv::rectangle(resultImage, rect, cv::Scalar(255,0,0));

					cv::Mat mnistImage = grayImage(rect);
					mnistImage = 255 - mnistImage;

					#ifdef TEST_MODE
						cv::imwrite("mnistImage.png", mnistImage);
					#endif

					double min, max;
					cv::minMaxLoc(mnistImage, &min, &max);

					cv::Size classRectSize = cv::Size(28,28);
					cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1/(max-min), classRectSize, cv::Scalar(min));
					
					m_net.setInput(blob);

					cv::Mat output = m_net.forward();
					

					int maxDigit = 0;
					double maxVal = output.at<float>(0,0);

					for (int i = 1; i < output.cols; i++){
						if (maxVal < output.at<float>(0,i)){
							maxVal = output.at<float>(0,i);
							maxDigit = i;
						}
					}
					
					std::string strVal = std::to_string(maxDigit);

					putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop, topLeftyCrop-5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0), 2);



				}
			}

			*m_proc_image[0] = resultImage;
		#else
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			int minSize = 20;
			int maxSize = 50;

			cv::Mat resultImage = colorImage.clone();
			cv::Mat resultImageContours = colorImage.clone();

			for(unsigned int idx = 0; idx < contours.size(); idx++){
				//area
				double area = cv::contourArea(contours[idx]);
				//bounding rectangle
				cv::Rect rect = cv::boundingRect(contours[idx]);
				//center of mass
				cv::Moments moment = cv::moments(contours[idx]);
				double cx = moment.m10 / moment.m00;
				double cy = moment.m01 / moment.m00;

				//draw contours at result image contours
				cv::drawContours(resultImageContours, contours, idx, cv::Scalar(255, 0, 0), 1, 8);
				
				int width = rect.width;
				int heigth = rect. height;
				int topLeftx = rect.x;
				int topLefty = rect.y;

				//ZaK: avoid duplicate code sections: the following part is also in block 
				//#ifdef REGION_LABELING above
				if((minSize <= width || minSize <= heigth) && width < maxSize && heigth < maxSize){
					int sizeCrop = (13*std::max(width, heigth))/10;
					int topLeftxCrop = std::max(0, topLeftx + (width - sizeCrop)/2);
					int topLeftyCrop = std::max(0, topLefty + (width - sizeCrop)/2);

					int widthCrop = std::min(sizeCrop, binaryImage.cols - topLeftxCrop);
					int heigthCrop = std::min(sizeCrop, binaryImage.rows - topLeftyCrop);

					cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heigthCrop);

					cv::rectangle(resultImage, rect, cv::Scalar(255,0,0));

					cv::Mat mnistImage = grayImage(rect);
					mnistImage = 255 - mnistImage;

					double min, max;
					cv::minMaxLoc(mnistImage, &min, &max);

					cv::Size classRectSize = cv::Size(28,28);
					cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1/(max-min), classRectSize, cv::Scalar(min));
					
					m_net.setInput(blob);

					cv::Mat output = m_net.forward();
					

					int maxDigit = 0;
					double maxVal = output.at<float>(0,0);

					for (int i = 1; i < output.cols; i++){
						if (maxVal < output.at<float>(0,i)){
							maxVal = output.at<float>(0,i);
							maxDigit = i;
						}
					}
					
					std::string strVal = std::to_string(maxDigit);

					putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop, topLeftyCrop-5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0), 2);

				}	

			}

			*m_proc_image[0] = resultImage;
			*m_proc_image[1] = resultImageContours;
			*m_proc_image[2] = binaryImage;
			
		
				
		#endif
			
			#ifdef TEST_MODE
				#ifndef RASPBERR_PI		//if not def!
					cv::imwrite("resultImage.png", resultImage);
				#endif
			#endif
	
	return(SUCCESS);
}









