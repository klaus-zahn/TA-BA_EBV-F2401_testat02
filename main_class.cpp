
#include "main_class.h"
#include "version.h"
#include "ipc.h"
// Exection Time
#include "time.h"


#include <unistd.h>


CMain::CMain() {
}


CMain::~CMain() {
	OscDestroy();
}



OSC_ERR CMain::Init(int argc, char ** argv) {
	
	OSC_ERR err;
	

	/******* Create the framework **********/
	if((err=OscCreate(
			&OscModule_log, 
			&OscModule_sup, 
			&OscModule_bmp, 
			&OscModule_cam,  
			&OscModule_vis
			//,&OscModule_gpio
			))!=SUCCESS)
		return(err);
	
	if(argc == 1) {
            OscLogSetConsoleLogLevel(NOTICE);
        } else {
            enum EnOscLogLevel level = (EnOscLogLevel) atol(argv[1]);
            printf("setting log level to %d", level);
            OscLogSetConsoleLogLevel(level);
        }
	OscLogSetFileLogLevel(WARN);
	
	//OscGpioConfigSensorLedOut(true);
	
#if defined(OSC_HOST) || defined(OSC_SIM)
	{
		void * hFileNameReader;
		
		if((err=OscFrdCreateConstantReader(&hFileNameReader, TEST_IMAGE_FN))!=SUCCESS)
			return(err);
		if((err=OscCamSetFileNameReader(hFileNameReader))!=SUCCESS)
			return(err);
	}
#endif /* OSC_HOST or OSC_SIM */
	

	/* init the camera */
	if((err=m_camera.Init(ROI(), 3))!=SUCCESS)
		return(err);
	
	m_camera.setAutoExposure(true);
	
	
	char* osc_version;
	if(OscGetVersionString(&osc_version) == SUCCESS) {
            string welcome_msg="###  "APP_NAME" "+getAppVersion().toStr()+"  OSCAR "+osc_version+"  ###\n";
            OscLog(INFO, welcome_msg.c_str());
        }
	
	
	return(SUCCESS);
}

OSC_ERR CMain::MainLoop() {
	OSC_ERR err=SUCCESS;
	
	CIPC ipc(m_camera, m_img_process);
	err=ipc.Init();
        
        /* do all init stuff here */
        OscCamSetShutterWidth(0);
        
        OscSimInitialize();
	
	/* read one image ahead */
	m_camera.CapturePicture();
	m_camera.ReadPicture();
        
	
	printf("read\n");
	
	uint32 startCyc=OscSupCycGet();
	
	while(err==SUCCESS) { /* infinite loop if no error occurs */
                /* read current picture and capture next */
			clock_t tStart = clock();
                uint32 startCycProc=OscSupCycGet();
		//ZaK: this statement does not "see" the definition in image_processing.cpp
		#ifndef TEST_MODE
			cv::Mat* img=m_camera.ReadPicture();
		#endif
                uint32 delta_time_us_proc=OscSupCycToMicroSecs(OscSupCycGet() - startCycProc);
                OscLog(DEBUG, "Image acquisition required %ums\n", delta_time_us_proc/1000);
                
                startCycProc=OscSupCycGet();
		OSC_ERR e=m_camera.CapturePicture();
                delta_time_us_proc=OscSupCycToMicroSecs(OscSupCycGet() - startCycProc);
                OscLog(DEBUG, "CapturePicture %ums\n", delta_time_us_proc/1000);
		
		if(e!=SUCCESS) OscLog(ERROR, "Could not Capture Picture (Error=%i)", e);
                
                startCycProc=OscSupCycGet();
                m_img_process.DoProcess(img);
                delta_time_us_proc=OscSupCycToMicroSecs(OscSupCycGet() - startCycProc);
                OscLog(DEBUG, "Image processing required %ums\n", delta_time_us_proc/1000);
		
                startCycProc=OscSupCycGet();
		err=ipc.handleIpcRequests();
                delta_time_us_proc=OscSupCycToMicroSecs(OscSupCycGet() - startCycProc);
                OscLog(DEBUG, "IPC request required %ums\n", delta_time_us_proc/1000);
		
		/* Allow other processes to run. Due to kernel tick rate of 250Hz this
		 * call will return in 4ms. */
		usleep(500);
		
		/* Advance the simulation step counter. */
		OscSimStep();
		
		
		uint32 delta_time_us=OscSupCycToMicroSecs(OscSupCycGet() - startCyc);
		if(delta_time_us/1000 > 1000) {
			startCyc=OscSupCycGet();
			if(ipc.img_count > 0)
				OscLog(DEBUG, "Sent %i images in %i ms\n", ipc.img_count, (int) delta_time_us/1000);
			ipc.img_count=0;
		}
	
		std::cout << "Time taken: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << std::endl;
	}
	return(err);
}



