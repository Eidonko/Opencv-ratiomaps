#include <string.h>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>              // std::pair
#include <iomanip>              // std::setprecision
#include <limits>               // std::numeric_limits
#include <vector>
#include <algorithm>    // std::max
#include <queue>

// OpenCV includes
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;

double Ps = 0.95;

void fromRGB2Mat(Mat& img, unsigned char **r, unsigned char **g, unsigned char **b);
void fromMat2RGB(Mat& img, unsigned char **r, unsigned char **g, unsigned char **b);
bool rowIsAllZeroes(int n, double* buf) ;
void createGnuplotHistogramWithCPD(string filename, int *histogram, int nelem, int& tin);

void badswitch(int c) {
	fprintf(stderr, "Bad switch: %c\n", c);
	exit(1);
}

bool MiniMax(unsigned char const *r, int rocols, int *rMin, int *rMax)
{
  int min=256;
  int max= -1;
  while (rocols > 0)
  {
	  rocols--;
	  if ( r[rocols] < min ) min = r[rocols];
	  if ( r[rocols] > max ) max = r[rocols];
  }
  *rMin = min, *rMax = max;
  return true;
}

void fromRGB2Mat(Mat& img, unsigned char **r, unsigned char **g, unsigned char **b) {
	for(int y = 0; y < img.rows; y++){
		for(int x = 0; x < img.cols; x++){
			Vec3b& intensity = img.at<Vec3b>(y, x);
			uchar& blu = intensity.val[0];
			uchar& gre = intensity.val[1];
			uchar& red = intensity.val[2];

			blu = b[y][x];
			gre = g[y][x];
			red = r[y][x];
		}
	}
}

void fromMat2RGB(Mat& img, unsigned char **r, unsigned char **g, unsigned char **b) {
	for(int y = 0; y < img.rows; y++){
		for(int x = 0; x < img.cols; x++){
			Vec3b& intensity = img.at<Vec3b>(y, x);
			uchar& blu = intensity.val[0];
			uchar& gre = intensity.val[1];
			uchar& red = intensity.val[2];

			b[y][x] = blu;
			g[y][x] = gre;
			r[y][x] = red;
		}
	}
}

inline void badSwitchInRatioMap(char c) {
	fprintf(stderr, "%c: Unknown switch in component cRatioMap.", c);
	fprintf(stderr, "Valid switches: -v(erbose)");
	fprintf(stderr, "                -i  inputGrid");
	fprintf(stderr, "                -oh output-HoI-Grid");
	fprintf(stderr, "                -os output-SoI-Grid");
	fprintf(stderr, "                -g(nuplotFile) path");
	fprintf(stderr, "                -a(djust) displacement");
	fprintf(stderr, "                -h(istogramFile) path");
}

/****************** R A T I O M A P S *******************/
//--------------------------------------------------------
int main(int argc, char *argv[]) {
//--------------------------------------------------------
/********************************************************/
	bool theSuccess;
	int verbose = 0;
	unsigned int options;
	int adjust = 0;
	int htin, stin;

	options = 0;

	string oSatFileName; // output grid 1
	string oHueFileName; // output grid 2
	string iFileName; // input grid
	string gFilePrefix; // Gnuplot output files' prefix
	string hFilePrefix; // Histogram files' prefix
	string tFileName; // clock ticks

	Mat src;

	// -v(Verbose), -i inputGrid, -oh Hue-over-Intensity-Ratiomap-Grid, -os Saturation-over-Intensity-Ratiomap-Grid
	// manage command-line args
	if (argc > 1)
		for (int i = 1; i < argc; i++)
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v':	verbose = 1; break;
				case 'o':
						if (strlen(argv[i]) == 2) {
							fprintf(stderr, "Invalid option: -o should be followed by either 's' or 'h'\n");
							return false;
						}
						switch (argv[i][2]) {
						case 's': case 'S':
							if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
							oSatFileName = argv[++i]; break;
						case 'h': case 'H':
							if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
							oHueFileName = argv[++i]; break;
						default:
							badSwitchInRatioMap(argv[i][2]);
							return false;
						}

						break;

				case 'i':	if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
						iFileName = argv[++i]; break;
				case 'a':	if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
						adjust = std::stoi(argv[++i]);
						break;
				case 'g':	if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
						gFilePrefix = argv[++i]; break;
				case 'h':	if (i >= argc - 1) {
								badSwitchInRatioMap(argv[i][1]);
								return false;
							}
						hFilePrefix = argv[++i]; break;
				default:
					badSwitchInRatioMap(argv[i][1]);
					return false;
				}
			}

	if (oSatFileName.empty()  &&  oHueFileName.empty() ) {
		fprintf(stderr, "ratioMaps: at least one out of \"-oh\" or \"-os\" must be present.\n");
		return false;
	}

	if (gFilePrefix.empty()  ||  hFilePrefix.empty() ) {
		fprintf(stderr, "ratioMaps: \"-g\" and \"-h\" must be present.\n");
		return false;
	}

	std::cerr << "Opening input file " << iFileName << '\n';
        src = imread(iFileName, cv::IMREAD_COLOR);
        namedWindow("Input image", CV_WINDOW_AUTOSIZE );
        imshow("Input image", src );

	const int rows = src.rows;
	const int cols = src.cols;

	std::cout << "Image " << iFileName << " consists of " << src.channels() << " channels and "
                << src.cols << "x" << src.rows << " pixels.\n";

	// we need to access the H, S, and I band at the same time => three row buffers are needed
        unsigned char *ph, *ps, *pi;

        unsigned char **h, **s, **i;
        unsigned char *buffh, *buffs, *buffi;

        int rocol = rows*cols;

        h = new unsigned char *[rows];
        s = new unsigned char *[rows];
        i = new unsigned char *[rows];

        buffh = new unsigned char[ rocol ];
        buffs = new unsigned char[ rocol ];
        buffi = new unsigned char[ rocol ];

        for (int idx=0, disp=0; idx<rows; idx++, disp += cols)
                h[idx] = & buffh[disp];
        for (int idx=0, disp=0; idx<rows; idx++, disp += cols)
                s[idx] = & buffs[disp];
        for (int idx=0, disp=0; idx<rows; idx++, disp += cols)
                i[idx] = & buffi[disp];

        fromMat2RGB(src, h, s, i);



	unsigned char *dstHoIdata = new unsigned char[rocol];
	unsigned char *dstSoIdata = new unsigned char[rocol];
	Mat dstHoI(rows, cols, CV_8UC1, dstHoIdata);
	Mat dstSoI(rows, cols, CV_8UC1, dstSoIdata);

	unsigned char *pH, *pS, *pI;
	unsigned char *pHM, *pSM;

	// frequencies of the values in the Ratio Maps
	int theHistogramHoIRatioMap[257];
	int theHistogramSoIRatioMap[257];

	// sum of the frequencies
	int  theTotalInHoIRatioMap = 0;
	int  theTotalInSoIRatioMap = 0;

	int theHoIVal, theSoIVal;
	vector<int> theHoIRow, theSoIRow;

	// Initialize histograms [[iH]]
	for (int l = 0; l < 257; l++)
		theHistogramHoIRatioMap[l] = theHistogramSoIRatioMap[l] = 0;

	int singularities = 0;


	for (int r = 0, disp=0; r < rows; r++, disp += cols) {
		pH = h[r], pS = s[r], pI = i[r];

		pHM = &dstHoIdata[disp], pSM = &dstSoIdata[disp];

		for (int c = 0; c < cols; c++, pH++, pS++, pI++) {
			if (! oHueFileName.empty() ) {
				if (*pI != 0) {
					//fprintf(debug, "isPixel ok for Hue and Intensity\n"); fflush(debug);
					// check for singularities
					if ((*pH + *pI + *pS < 3) || (*pH == *pS && *pS == *pI)) {
						*pHM = 0; // record the singularity in the ratio map

						singularities++;
					}
					else {
						//fprintf(debug, "%d,%d: non singularity case\n", r, c); fflush(debug);

						try {
							*pHM = int(round((float) *pH / (*pI + 1)));
						}
						catch (std::exception& e) {
							fprintf(stderr, "Exception caught while computing the hue-over-intensity Ratio Map: %s\n", e.what());
							return false;
						}
						theHoIVal = *pHM;
						if (theHoIVal >= 0 && theHoIVal < 256) {
							theHistogramHoIRatioMap[theHoIVal]++;
							theTotalInHoIRatioMap++;
						}
						else { // saturated values go in entry 256
							theHistogramHoIRatioMap[256]++;
							theTotalInHoIRatioMap++;
						}
					}
					pHM++;
				}
				else { // isPixel() == false
					*pHM = 0;
					pHM++;
				}
			}

			if (! oSatFileName.empty() ) {
				if (! *pI) {
					try {
						*pSM = int(round((float)*pS / (*pI + 1)));
					}
					catch (std::exception& e) {
						fprintf(stderr, "Exception caught while computing the saturation-over-intensity Ratio Map: %s\n", e.what());
						return false;
					}

					theSoIVal = *pSM;

					if (theSoIVal >= 0 && theSoIVal < 256) {
						theHistogramSoIRatioMap[theSoIVal]++;
						theTotalInSoIRatioMap++;
					} else {
						theHistogramSoIRatioMap[256]++;
						theTotalInSoIRatioMap++;
					}
				}
				else {
					*pSM = 0; //  -1; // theGridDesc.noData;
				}
				pSM++;
			}
		} // end column loop
	} // end row loop

	fprintf(stderr, "%d singularities were found (%lf%%)\n", singularities, (double)singularities / (rows*cols) * 100.0);

	fprintf(stderr, "gFilePrefix == %s, hFilePrefix == %s\n", gFilePrefix.c_str(), hFilePrefix.c_str());

	if (! gFilePrefix.empty() ) {
		if (! oHueFileName.empty() ) createGnuplotHistogramWithCPD(gFilePrefix + "-HoI", theHistogramHoIRatioMap, 257, htin);
		if (! oSatFileName.empty() ) createGnuplotHistogramWithCPD(gFilePrefix + "-SoI", theHistogramSoIRatioMap, 257, stin);
	}

	fprintf(stderr, "HoI threshold index is %d. SoI threshold index is %d.\n", htin, stin);
	fprintf(stderr, "\t=>   XML configurations written in files \"SMask.H.xml\" and \"SMask.S.xml\"\n");
//	GSLOG_USR_INFO("\n\tre-map-input=\"%s\"\n\tre-map-equivalence-classes=\"%d 255 1\"\n\tre-map-default=\"0\"\n\tre-map-output=\"remapped-%s\"\n",
//		oHueFileName.c_str(), htin, oHueFileName.c_str());
//	GSLOG_USR_INFO("\n\tfelzen-segm-input=\"%s\"\n\tfelzen-segm-output=\"%s\"\n\tfelzen-segm-sigma=\"0.5\"\n\tfelzen-segm-kappa=\"1000\"\n\tfelzen-segm-minsize=\"250\"\n",
//		("remapped-" + oHueFileName).c_str(), ("felzenszwalb-remapped-" + oHueFileName).c_str());
//
//	GSLOG_USR_INFO("\n\tre-map-input=\"%s\"\n\tre-map-equivalence-classes=\"%d 255 1\"\n\tre-map-default=\"0\"\n\tre-map-output=\"remapped-%s\"\n",
//		oSatFileName.c_str(), stin, oSatFileName.c_str());
//	GSLOG_USR_INFO("\n\tfelzen-segm-input=\"%s\"\n\tfelzen-segm-output=\"%s\"\n\tfelzen-segm-sigma=\"0.5\"\n\tfelzen-segm-kappa=\"1000\"\n\tfelzen-segm-minsize=\"250\"\n",
//		("remapped-" + oSatFileName).c_str(), ("felzenszwalb-remapped-" + oSatFileName).c_str());

	string minSize = "250";

	{
		FILE *f;
		f = fopen("SMask.H.xml", "w");
		if (f == NULL) {
			fprintf(stderr, "Could not open XML output file -- sending output to stderr instead.\n");
			f = stderr;
		}
		if (htin > 0) htin--;

		fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
		fprintf(f, "<SMask xmlns=\"http://www.vito.be/CDPC/SMask\">\n");
		fprintf(f, "\t<Grids>\n\t\t<Grid\n");
		fprintf(f, "\t\tinput=\"%s\"\n", oHueFileName.c_str());
		fprintf(f, "\t\tmethod=\"remap|felzenszwalb|segmentor\"\n");
		fprintf(f, "\n\t\tre-map-input=\"%s\"\n\t\tre-map-equivalence-classes=\"%d 255 1\"\n\t\tre-map-default=\"0\"\n\t\tre-map-output=\"remapped-%s\"\n",
			oHueFileName.c_str(), (htin + adjust >= 0)? htin + adjust : htin, oHueFileName.c_str());
		fprintf(f, "\n\t\tfelzen-segm-input=\"%s\"\n\t\tfelzen-segm-output=\"%s\"\n\t\tfelzen-segm-sigma=\"0.5\"\n\t\tfelzen-segm-kappa=\"1000\"\n\t\tfelzen-segm-minsize=\"%s\"\n",
			("remapped-" + oHueFileName).c_str(), ("felzenszwalb-remapped-" + oHueFileName).c_str(), minSize.c_str());
		fprintf(f, "\n\t\trle-segments-input=\"%s\"\n\t\trle-segments-output=\"%s\"\n\t\trle-segments-python=\"%s\"\n\t\trle-segments-min-area=\"10\"\n",
			("remapped-" + oHueFileName).c_str(), ("rle-remapped-" + oHueFileName).c_str(), ("rle-remapped-python" + oHueFileName).c_str());

		fprintf(f, "\t\t/>\n\t</Grids>\n</SMask>\n");
		if (f != stderr) fclose(f);
	}

	{
		FILE *f;
		f = fopen("SMask.S.xml", "w");
		if (f == NULL) {
			fprintf(stderr, "Could not open XML output file -- sending output to stderr instead.\n");
			f = stderr;
		}
		fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
		fprintf(f, "<SMask xmlns=\"http://www.vito.be/CDPC/SMask\">\n");
		fprintf(f, "<Grids>\n  <Grid\n");
		fprintf(f, "\tinput=\"%s\"\n", oSatFileName.c_str());
		fprintf(f, "\tmethod=\"remap|felzenszwalb|segmentor\"\n");
		fprintf(f, "\n\tre-map-input=\"%s\"\n\tre-map-equivalence-classes=\"%d 255 1\"\n\tre-map-default=\"0\"\n\tre-map-output=\"remapped-%s\"\n",
			oSatFileName.c_str(), (stin + adjust >= 0) ? stin + adjust : stin, oSatFileName.c_str());
		fprintf(f, "\n\tfelzen-segm-input=\"%s\"\n\tfelzen-segm-output=\"%s\"\n\tfelzen-segm-sigma=\"0.5\"\n\tfelzen-segm-kappa=\"1000\"\n\tfelzen-segm-minsize=\"%s\"\n",
			("remapped-" + oSatFileName).c_str(), ("felzenszwalb-remapped-" + oSatFileName).c_str(), minSize.c_str());
		fprintf(f, "\n\t\trle-segments-input=\"%s\"\n\t\trle-segments-output=\"%s\"\n\t\trle-segments-python=\"%s\"\n\t\trle-segments-min-area=\"10\"\n",
			("remapped-" + oSatFileName).c_str(), ("rle-remapped-" + oSatFileName).c_str(), ("rle-remapped-python" + oSatFileName).c_str());

		fprintf(f, "  />\n</Grids></SMask>\n");
		if (f != stderr) fclose(f);
	}

	FILE *maps;
	if (hFilePrefix.empty() )
		hFilePrefix = "histogram";

	if (! oHueFileName.empty() ) {
		string fname = hFilePrefix + "-HoI.rmh";
		maps = fopen(fname.c_str(), "w");
		if (maps == NULL) {
			fprintf(stderr, "Failed to open output ratio-map histogram (hue over intensity)\n");
		}
		fprintf(maps, "RMAP_HISTOGRAM\n257\n");
		for (int i = 0; i < 257; i++) {
			fprintf(maps, "%d\n", theHistogramHoIRatioMap[i]);
		}
		fclose(maps);
	}
	if (! oSatFileName.empty() ) {
		string fname = hFilePrefix + "-SoI.rmh";
		maps = fopen(fname.c_str(), "w");
		if (maps == NULL) {
			fprintf(stderr, "Failed to open output ratio-map histogram (saturation over intensity)\n");
		}
		fprintf(maps, "RMAP_HISTOGRAM\n257\n");
		for (int i = 0; i < 257; i++) {
			fprintf(maps, "%d\n", theHistogramSoIRatioMap[i]);
		}
		fclose(maps);
	}

        namedWindow("H/I image", CV_WINDOW_AUTOSIZE );
        imshow("H/I image", dstHoI );
	imwrite(oHueFileName, dstHoI );

        namedWindow("S/I image", CV_WINDOW_AUTOSIZE );
        imshow("S/I image", dstSoI );
	imwrite(oSatFileName, dstSoI );

        delete h, delete s, delete i;
        delete buffh, buffs, buffi;
	delete dstHoIdata, dstSoIdata;

	waitKey(0);

	return true;
}

void createGnuplotHistogramWithCPD(string filename, int *histogram, int nelem, int& tin) {
	ofstream f;
	string s;
	string input, gnuplot;
	int sum; // , tin;
	int infimum;
	double threshold;

	input = filename + ".input.txt";
	f.open(input);
	for (int i = sum = 0; i < nelem; i++) {
		sum += histogram[i];
		f << histogram[i] << ' ' << sum << '\n';
	}
	f.close();

	threshold = sum * Ps;

	for (tin = infimum = 0; tin < nelem && infimum < threshold; tin++) {
		infimum += histogram[tin];
	}
	infimum -= histogram[tin];
	tin--;

	int mini = 0, maxi = 255;
	while (histogram[mini] == 0) mini++;
	while (histogram[maxi] == 0) maxi--;

	gnuplot = filename + ".gpt";
	f.open(gnuplot);
	const int sup = std::min(maxi, 50); // 40;
	f << "unset key\n" << "set style data histogram\n" << "set style fill solid border\n"
		<< "set xrange [-10:266]\n"
		<< "set xrange [0:" << sup << "]\n"
		// << "set xrange [" << mini << ':' << maxi << "]\n"
		<< "set title \"" << filename << "\"\n"
		<< "set style line 1 lt 1 lw 3 pt 7 linecolor rgb \"red\"\n"
		<< "set style line 2 lt 2 lw 3 pt 2 linecolor rgb \"blue\"\n"
		<< "set style line 3 lt 3 lw 5 pt 5 linecolor rgb \"green\"\n"
		//<< "set terminal pdf\n" << "set output '" << filename << ".pdf'\n"
		<< "set terminal pdf size 16in, 3in\n" << "set output '" << filename << ".pdf'\n"
		<< "set label \"Ts=" << tin << "\" at " << tin << ", " << sum*1.05 << '\n'
		<< "set label \"Inf=" << infimum << "\" at " << sup+1.3 << ", " << threshold * 0.8 << " rotate by 90\n"
		<< "set arrow from " << tin << ", " << sum*1.05 << " to " << tin << ", " << infimum << " ls 1\n"
		<< "plot '" << input << "' using 0:1 with imp ls 1, '' using 0:1 with lines ls 1, "
		<< "'' using 0:2 with imp ls 2, '' using 0:2 with lin ls 2, "
		<< threshold << " ls 3\n";

	//<< "plot \"" << filename + ".input.txt" << "\" using 0:1:(sprintf(\"(%d)\", $0, $1)) with labels point  pt 7 offset char 1 notitle\n";
	f.close();

	s = "gnuplot " + gnuplot;
	std::system(s.c_str());
}

// EoM.R A T I O M A P S

