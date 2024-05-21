

#include "image_processing.h"

//#define TEST_MODE ;


CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		#ifdef TEST_MODE 
			m_net = cv::dnn::readNetFromONNX("MnistCNN_Gray.onnx"); 
		#else
			m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx"); 
		#endif
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
	
	if(!image) return(EINVALID_PARAMETER);	
        
        #ifdef TEST_MODE 
			*image = cv::imread("MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE); 
		#endif


        cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);

		if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		} else {
			grayImage = *image;
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}

		







		if (mPrevImage.size() != cv::Size()) {

			cv::Mat diffImage;
			cv::absdiff(*image, mPrevImage, diffImage);

			cv::Mat binaryImage;
			cv::threshold(diffImage, binaryImage, threshold, 255,
			cv::THRESH_BINARY);

			cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
			cv::morphologyEx(binaryImage, binaryImage,
			cv::MORPH_CLOSE, kernel);

			cv::Mat stats, centroids, labelImage;
			connectedComponentsWithStats(binaryImage, labelImage,
			stats, centroids);

			cv::Mat resultImage = image->clone();

			for (int i = 1; i < stats.rows; i++) {

				int topLeftx = stats.at<int>(i, 0);
				int topLefty = stats.at<int>(i, 1);
				int width = stats.at<int>(i, 2);
				int height = stats.at<int>(i, 3);
				int area = stats.at<int>(i, 4);
				double cx = centroids.at<double>(i, 0);
				double cy = centroids.at<double>(i, 1);

				

				cv::Rect rect(topLeftx, topLefty, width, height);
				cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
				cv::Point2d cent(cx, cy);
				cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);

				

				}

			if (init == 0) {
				mBkgrImage = grayImage.clone();
				init = 1;
			}
			double alpha = 0.9;
			cv::addWeighted(mBkgrImage, alpha, grayImage, 1 - alpha, 0, mBkgrImage);

			diffBkgrImage = (255+ mBkgrImage - grayImage)/2;
			
			cv::threshold(diffBkgrImage, binaryImage, threshold, 255,
			cv::THRESH_BINARY);

		
			std::vector<std::vector<cv::Point> > contours;
 			std::vector<cv::Vec4i> hierarchy;

			cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL ,
			cv::CHAIN_APPROX_SIMPLE);

			for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
			//area
			double area = cv::contourArea(contours[idx]);
			//bounding rectangle
			cv::Rect rect = cv::boundingRect(contours[idx]);
			//center of gravity
			// center of mass
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;
			//to draw counter to index idx in image
			cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );
			}

			int minSize = 20;
			int maxSize = 40;

			double threshold1 = 50; 
			double threshold2 = 200;
			cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);
			kernel = cv::Mat::ones(5, 5, CV_8UC1); cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);
			stats, centroids, labelImage; 
			//connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

			cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL ,
			cv::CHAIN_APPROX_SIMPLE);

			for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
			//area
			double area = cv::contourArea(contours[idx]);
			//bounding rectangle
			cv::Rect rect = cv::boundingRect(contours[idx]);
			//center of gravity
			// center of mass
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;
			//to draw counter to index idx in image
			cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );
			}

			resultImage = colorImage.clone();
		
			for (int i = 1; i < stats.rows; i++) {
				int topLeftx = stats.at<int>(i, 0); 
				int topLefty = stats.at<int>(i, 1); 
				int width = stats.at<int>(i, 2); 
				int height = stats.at<int>(i, 3);
				int area = stats.at<int>(i, 4);

				if((minSize <= width || minSize <= height) &&width < maxSize && height < maxSize) {

					int sizeCrop = (13*std::max(width, height))/10;
					int topLeftxCrop = std::max(0, topLeftx+(width-sizeCrop)/2); 
					int topLeftyCrop = std::max(0, topLefty+(height-sizeCrop)/2);
					int widthCrop = std::min(sizeCrop, binaryImage.cols-topLeftxCrop);
					int heightCrop = std::min(sizeCrop, binaryImage.rows-topLeftyCrop);
					cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heightCrop);

					cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

					cv::Mat mnistImage = grayImage(rect);
					mnistImage = 255 - mnistImage;

					#ifdef TEST_MODE
					cv::imwrite("mnistImage.png", mnistImage); 
					#endif

					double min, max; 
					cv::minMaxLoc(mnistImage, &min, &max);

					cv::Size classRectSize = cv::Size(28, 28);

					cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1./(max-min),classRectSize, cv::Scalar(min));

					m_net.setInput(blob); 
					cv::Mat output = m_net.forward();

					float maxProbability = output.at<float>(0, 0);

					for(int i0 = 0; i0 < output.cols; i0++) { 
						std::cout << i0 << "," << output.at<float>(0,i0) << std::endl;
						if (output.at<float>(0, i0) > maxProbability) {
        				maxProbability = output.at<float>(0, i0);
        				maxDigit = i0;
    					}
					}

					std::string strVal = std::to_string(maxDigit); 
					putText(resultImage, strVal.c_str(), cv::Point(topLeftxCrop,
					topLeftyCrop-5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);


					#ifdef TEST_MODE
						cv::imwrite("resultImage.png", resultImage); 
					#endif


				}

				

			}
			

			*m_proc_image[0] = resultImage;
			*m_proc_image[1] = imgCanny;
			*m_proc_image[2] = mnistImage;

		}
		mPrevImage = grayImage.clone();


		
        
      //  cv::imwrite("dx.png", *m_proc_image[0]);
      //  cv::imwrite("dy.png", *m_proc_image[1]);

	return(SUCCESS);
}









