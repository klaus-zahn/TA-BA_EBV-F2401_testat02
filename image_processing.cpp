
#include "image_processing.h"

///////////// TEST MODE ON  wenn auskommentiert dann off ////////////////////
//#define TEST_MODE
/////////////////////////////////////////////////////////////////////////////

// Konstruktor
CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
	// EInlesen des NN je nach modus (rpi oder vm)
		#ifdef TEST_MODE
			m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/model.onnx");
		#else 
			m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
		#endif
}

// Dekonstruktor
CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
}

// Methode

cv::Mat *CImageProcessor::GetProcImage(uint32 i)
{
	if (2 < i)
	{
		i = 2;
	}
	return m_proc_image[i];
}


// Methode
int CImageProcessor::DoProcess(cv::Mat *image)
{	
	cv::Mat binaryImage;
	cv::Mat imgCanny;
	cv::Mat grayImage;
	cv::Mat colorImage;
	int minSize=20;
	int maxSize=35;

	if (!image)
		return (EINVALID_PARAMETER);

	#ifdef TEST_MODE
		*image = cv::imread("/home/pi/CNNdir/MnistRaspiImage_c.png", cv::IMREAD_GRAYSCALE);
	#endif

	
	

	if (image->channels() > 1) {
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	} else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}
	
	
	double threshold1 = 50;
	double threshold2 = 200;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(imgCanny, binaryImage, cv::MORPH_DILATE, kernel);

	std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat resultImage = colorImage.clone();

	for (unsigned int idx = 0; idx < contours.size(); idx++) {

		cv::Rect rect = cv::boundingRect(contours[idx]);
        int topLeftx = rect.x;
        int topLefty = rect.y;
        int width = rect.width;
        int height = rect.height;


		if((minSize <= width || minSize <= height) && width < maxSize && height < maxSize) {
			
			int sizeCrop = (13*std::max(width, height))/10;
			int topLeftxCrop = std::max(0, topLeftx+(width-sizeCrop)/2);
			int topLeftyCrop = std::max(0, topLefty+(height-sizeCrop)/2);

			int widthCrop = std::min(sizeCrop, binaryImage.cols- topLeftxCrop);
			int heightCrop = std::min(sizeCrop, binaryImage.rows- topLeftyCrop);

			cv::Rect rect(topLeftxCrop, topLeftyCrop, widthCrop, heightCrop);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

			
			cv::Mat mnistImage = grayImage(rect);
			mnistImage = 255 - mnistImage;

			#ifdef TEST_MODE
				cv::imwrite("/home/pi/CNNdir/mnistImage.png", mnistImage); 
			#endif

			double min, max; 
			cv::minMaxLoc(mnistImage, &min, &max);

			cv::Size classRectSize = cv::Size(28, 28);
			cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1./(max-min),
												classRectSize, cv::Scalar(min));
			m_net.setInput(blob);
			cv::Mat output = m_net.forward();

			//ZaK: rather skip output for productive code
			for(int i0 = 0; i0 < output.cols; i0++) { 
				std::cout << i0 << "," << output.at<float>(0,i0) << std::endl; }

			cv::Point maxLoc;
            minMaxLoc(output.reshape(1, 1), nullptr, nullptr, nullptr, &maxLoc);
            int maxDigit = maxLoc.x;
			
			std::string strVal = std::to_string(maxDigit); 
			putText(resultImage, strVal.c_str(), 
					cv::Point(topLeftxCrop, topLeftyCrop-5), 
					cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, 
					cv::Scalar(255, 0, 0), 2);
		}
		
	}
	#ifdef TEST_MODE
		cv::imwrite("/home/pi/CNNdir/Result.png", resultImage); 
	#endif


	*m_proc_image[0] = resultImage;
	*m_proc_image[1] = imgCanny;
	*m_proc_image[2] = binaryImage;

	return (SUCCESS);
}
