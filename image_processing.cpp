/* das define TEST_MODE ist in der Zeile 31 */
#include "image_processing.h"


CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
	#ifdef TEST_MODE // will never be defined since i can only work on the raspberry pi itself (no ubuntu vm)
	m_net = cv::dnn::readNetFromONNX("MnistCNN_Gray.onnx");
	#else
	m_net = cv::dnn::readNetFromONNX("/home/pi/CNNdir/MnistCNN_Gray.onnx");
	// MNIST = Modified National Institute of Standards and Technology
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

// #define TEST_MODE // Funktioniert nicht mit "make run" sondern nur im debugger
#define SEGFAULT_WORKAROUND

/* !todo The following Error occurs, when TEST_MODE is defined and executing "make run":
terminate called after throwing an instance of 'cv::Exception'
what():  OpenCV(4.1.1) /home/ccisn/vca/vcageneric/trunk/externals/OpenCV4.1.1.linux.x86/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
-> Works when started with debugging tool 
*/

int CImageProcessor::DoProcess(cv::Mat* image) {
	if(!image) return(EINVALID_PARAMETER);	

	static cv::Mat colorImage, grayImage, binaryImage, resultImage, imgCanny, hierarchy; 
	static cv::Mat stats, centroids, labelImage;
	double threshold1 = 50, threshold2 = 200;
	int minSize = 20, maxSize = 40;
	cv::Rect rect;
	cv::Point maxDigit;

	#ifdef TEST_MODE //ZaK: filename is case sensitive 'M' not 'm' 
	*image = cv::imread("./MnistRaspiImage_c.png",cv::IMREAD_GRAYSCALE);
	// *image = cv::imread("../CNNdir/MnistRaspiImage_c.png",cv::IMREAD_GRAYSCALE); // funktioniert nicht obwohl Bild auch dort
	#endif

	// Zur Abspeicherung von s/w- und farbbilder 
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY); // konvertiere zu s/w, falls das Bild farbig ist
		colorImage = image->clone();
	} else {
		grayImage = *image;	// speicher das Bild ab falls es schon s/w ist
		cv::cvtColor(*image,colorImage,cv::COLOR_GRAY2RGB); // konvertiere das s/w bild zu rgb für die Anzeige im Web 
	}

	cv::Canny(grayImage, imgCanny ,threshold1, threshold2, 3, true); // Edge Detection
	cv::Mat kernel = cv::Mat::ones(5,5,CV_8UC1); // Structuring Element
	cv::morphologyEx(imgCanny,binaryImage,cv::MORPH_DILATE, kernel);


	// connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage, contours, hierarchy,cv::RetrievalModes::RETR_EXTERNAL,cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	
	resultImage = colorImage.clone();
	for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
		cv::Rect rect = cv::boundingRect(contours[idx]); // bounding rectangle
		
		cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 ); // to draw contour to index idx in image

		int topLeftx = rect.x;
		int topLefty = rect.y;
		int width = rect.width;
		int height = rect.height;

		if((width >= minSize || height >= minSize)&&(width <= maxSize && height <= maxSize)){
			int cropSize = (13*std::max(width,height))/10; // multipliziere um Faktor 1.3 wegen Rand
			int topLeftxCrop = std::max(0,topLeftx+(width-cropSize)/2); 
			int topLeftyCrop = std::max(0,topLefty+(height-cropSize)/2); 
			
			int widthCrop = std::min(cropSize, binaryImage.cols-topLeftxCrop); // Um Sicherzustellen, dass die Teilregionen vollständig im Bild liegen
			int heightCrop = std::min(cropSize, binaryImage.rows-topLeftyCrop); 

			rect = cv::Rect(topLeftxCrop,topLeftyCrop,widthCrop,heightCrop);
			cv::rectangle(resultImage,rect,cv::Scalar(255,0,0)); // draws the rectangle onto the resultImage

			// Schneide das Rechteck aus (und speichere es ab)
			cv::Mat mnistImage = 255-grayImage(rect);

			#ifdef TEST_MODE
			cv::imwrite("./mnistImage.jpg",mnistImage);	
			#endif

			double min, max; // min/max Location and Value of image 
			cv::minMaxLoc(mnistImage, &min, &max);

			// Classification
			cv::Size classRectSize = cv::Size(28,28);
			cv::Mat blob = cv::dnn::blobFromImage(mnistImage, 1./(max-min),classRectSize,cv::Scalar(min));


			#ifdef SEGFAULT_WORKAROUND
			cv::Mat output = cv::Mat::zeros(1, 10, CV_32F);;
			#else
			m_net.setInput(blob); /*!todo SEGMENTATION FAULT */
			cv::Mat output = m_net.forward();
			#endif

			// Find number with highest soft-max probability and print it onto resultImage
			cv::minMaxLoc(output,NULL,NULL,NULL,&maxDigit);
			std::string strVal = cv::format("%d", maxDigit.x); // !todo kann nicht kontrollieren ob .x oder .y benutzt werden muss  
			cv::putText(resultImage,strVal.c_str(), cv::Point(topLeftxCrop,topLeftyCrop -5),cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2);
		}
	}
	
	// Auswahl der Bildverarbeitungsmethoden (im web)
	*m_proc_image[0] = resultImage.clone();
	*m_proc_image[1] = imgCanny.clone();
	*m_proc_image[2] = binaryImage.clone();


	return(SUCCESS);
}







