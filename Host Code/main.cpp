#include <iostream>
#include <string>

#include <fstream>
// Profilling file
std::ofstream profilling_file("Profilling_info.txt");



#define type_time std::chrono::high_resolution_clock::time_point
#define type_time_calc std::chrono::duration<double>

int counter = 0;

#include "phase_eulerian.hpp"


int main(int argc, char* argv[]){

		// TARGET_DEVICE macro needs to be passed from gcc command line
		if (argc != 2) {
			std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
			return EXIT_FAILURE;
		}

		std::string xclbinFilename = argv[1];

		int maxFrames = 60;
		int numberOfFrames = 60; //60


		int height = 100; //100;
		int WIDTH = 300; //300; // here *3
	    double alpha;
	    double cutoff_freq_low;
	    double cutoff_freq_high;
	    double sigma;
	    bool attenuateOtherFreq;
	    std::string pyrType = "octave";
        std::string temporalFilter = "differenceOfIIR";


	    attenuateOtherFreq = false;
	    alpha = 100.0;
	    cutoff_freq_low = 1.0;
	    cutoff_freq_high = 3.0;
	    sigma = 0;

	    //KR260 path -> /home/petalinux/home/data/data/video_frame_data_in/
	    //KR260 path -> /home/petalinux/home/data/data/video_frame_data_out/

       //std::string input_path = "/media/sheldon/50246327-c3ad-4255-a51a-664a42f7eacd/ngiannopoulos/kria_tutorial/Lria_KR260/kr260_custom_platform/data folder/data/video_frame_data_in";
	   //std::string output_path = "/media/sheldon/50246327-c3ad-4255-a51a-664a42f7eacd/ngiannopoulos/kria_tutorial/Lria_KR260/kr260_custom_platform/data folder/data/video_frame_data_out";

	    std::string input_path = "/home/petalinux/home/phase_transf_folder/data/data/video_frame_data_in/";
	    std::string output_path = "/home/petalinux/home/phase_transf_folder/data/data/video_frame_data_out/";



	    std::vector<std::vector<std::vector<uint8_t>>> vid(numberOfFrames, std::vector<std::vector<uint8_t>>(height,std::vector<uint8_t>(WIDTH)));
	    std::vector<std::vector<std::vector<uint8_t>>> vid_out(numberOfFrames, std::vector<std::vector<uint8_t>>(height,std::vector<uint8_t>(WIDTH)));

	    for(counter; counter < 100; counter++){

	    for(int k=0; k < maxFrames; k++){
	    	std::string filename = input_path + "video_frame_" + std::to_string(k) + ".txt";
	    	std::ifstream file(filename);
	    	if (!file.is_open()) {
	    		std::cerr << "Error opening file: " << filename << std::endl;
	    	}

	    	for(int i = 0; i<height; ++i){
	    		std::string line;
	    		if(!std::getline(file,line)){
	    			std::cerr << "Error read from file: " << filename << std::endl;
	    		}
	    		std::istringstream line_stream(line);

	    		for(int j = 0; j < WIDTH; ++j){
	    			int value;
	    			if(!(line_stream >> value)){
	    				std::cerr << "Error reading RGB data: " << filename << std::endl;
	    			}

	    			vid[k][i][j] = static_cast<uint8_t>(value);
	    		}
	    	}
	    	file.close();
	    }


	    type_time all = std::chrono::high_resolution_clock::now();

	    phase_eulerian( xclbinFilename,
	    		vid,
				alpha,
				cutoff_freq_low,
				cutoff_freq_high,
				vid_out, // fianl results out
				sigma,
				attenuateOtherFreq,
				pyrType,
				temporalFilter);
	    type_time end = std::chrono::high_resolution_clock::now();
	    type_time_calc all_duration = end - all;
	    profilling_file << "All Execution time "<< all_duration.count() << " seconds." << std::endl;

	    for(int k=0; k < maxFrames; k++){
	    	std::string filename_out = output_path + "video_frame_" + std::to_string(k) +"_final.txt";

	    	// Open a new file for writing the vidData
	    	std::ofstream file_out(filename_out);
	    	if (!file_out.is_open()) {
	    		std::cerr << "Error opening file for writing: " << filename_out << std::endl;
	    	}


	    	for (int i = 0; i < height; i++) {
	    		for(int j=0; j <WIDTH / 3; j++){
	    			if(j == (height-1)){

	    				file_out << static_cast<int>(vid_out[k][i][j*3]) << " " << static_cast<int>(vid_out[k][i][j*3 + 1]) << " "
																								<< static_cast<int>(vid_out[k][i][j*3+2]);

	    			}else{

	    				file_out << static_cast<int>(vid_out[k][i][j*3]) << " " << static_cast<int>(vid_out[k][i][j*3 + 1]) << " "
																				<< static_cast<int>(vid_out[k][i][j*3+2]) << " ";
	    			}
	    		}
	    		file_out << std::endl;
	    	}

	    	file_out.close();

	    }
	    }

	    std::cout <<"Done\n";
	    return 0;
	    exit(0);
}



