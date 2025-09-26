
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <filesystem>
#include <mpi.h>
#include "String.h"
#include "Vector3.hpp"
#include "Quaternion.hpp"
#include "ml.hpp"
#include "wrapper.hpp"
#include "misc.hpp"

using namespace std;
using namespace filesystem;

int main(int argc, char* argv[])
{
  int nproc, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  string os_file = "op_settings.json";
  int fnum[3] = {-1, -1, -1};
  string header = "";
  string data_frame = "data_frame";

  for(int i = 1; i < argc; ++i) {
    string s = argv[i];
    if( (s == "-os" || s == "--op_settings") && i + 1 < argc ) {
      os_file = argv[i+1];
      ++i;
      continue;
    }
    if( (s == "-fi" || s == "--filenumber") && i + 3 < argc ) {
      fnum[0] = atoi(argv[i + 1]);
      fnum[1] = atoi(argv[i + 2]);
      fnum[2] = atoi(argv[i + 3]);
      i += 3;
      continue;
    }
    if( (s == "-h" || s == "--header") && i + 1 < argc ) {
      header = argv[i+1];
      ++i;
      continue;
    }
    if( (s == "-df" || s == "--data_frame") && i + 1 < argc ) {
      data_frame = argv[i+1];
      ++i;
      continue;
    }
  }

  if( os_file == "" || fnum[0] < 0 || fnum[1] < 1 || fnum[2] < 1 ||
      data_frame == "" ) {
    cerr << "Usage: " << argv[0] << " -os OP_SETTINGS_JSON "
         << "-fi FILE_BEGIN FILE_NUMBER FILE_INTERVAL -h HEADER "
         << "-df DATA_FRAME_PREFIX"
         << endl;
    exit(1);
  }

  vector<String> headers = String(header).Split(",");

  OP_SETTINGS op_settings;
  op_settings.ReadJson(os_file);
  
  int bnum = fnum[0];
  int snum = fnum[1];
  int dnum = fnum[2];
  vector<int> dlist{0};
  //vector<int> dlist{0,5,10,15,20,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24};

  if( rank == 0 )
    cout << "For debug, dnum, len(dlist): " << dnum << " " << dlist.size() << endl;

  vector<string> filename_list_r;
  vector<string> filename_list_p;
  for( int i : dlist ) {
    for( int j = 0; j < snum; ++j ) {
      ostringstream oss;
      oss << std::setw(7) << std::setfill('0') << i + bnum + j * dnum;
      string s = "__input/" + oss.str() + ".txt";
      filename_list_r.push_back(s);
      filename_list_p.push_back(oss.str());
    }
  }

  if( rank == 0 ) {
    cout << "Total number of processing SS    : " << filename_list_r.size() << endl;
  
    path tmpdir = "__tmp";
    if (!exists(tmpdir)) {
      create_directory(tmpdir);
    }
  }

  // Loop for LOPs calculation start #
  clock_t t1 = clock();

  int n10 = 10;
  for(int n = rank; n < (int)filename_list_r.size(); n += nproc ) {

    const string& file_name = filename_list_r[n];

    //# Write elapsed time #
    if( rank == 0 && n > n10 ) {
      clock_t t2 = clock();
      double time = double(t2 - t1) / CLOCKS_PER_SEC;
      cout << "### Record ### Comput.OP:" << n <<  "[times]" << endl;
      cout << "### Record ### Elap.Time:" << time << "[sec]" << endl;
      n10 += 10;
    }

    //# Read LAMMPS coordinates #
    ifstream fin(file_name.c_str());
    if( fin.fail() ) {
      cerr << "ERROR: Failed to read " << file_name << endl;
    }
    string line;

    int nmol = 0;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    vector<double> posx, posy, posz, qua1, qua2, qua3, qua4;
    vector<int> mid;
    int iid = -1, ix = -1, iy = -1, iz = -1, iq1 = -1, iq2 = -1, iq3 = -1, iq4 = -1;
    bool bQuat = false;
    while (getline(fin, line)) {
      String l = String(line).Trim();
      vector<String> words = l.Split();
      if( l == "ITEM: NUMBER OF ATOMS" ) {
        getline(fin, line);
        nmol = String(line).ToInt();
        continue;
      }
      if( l.substr(0, 16) == "ITEM: BOX BOUNDS" ) {
        getline(fin, line);
        words = String(line).Split();
        lx = words[1].ToDouble();

        getline(fin, line);
        words = String(line).Split();
        ly = words[1].ToDouble();
          
        getline(fin, line);
        words = String(line).Split();
        lz = words[1].ToDouble();
        continue;
      }
      if( l.substr(0, 11) == "ITEM: ATOMS" ) {
        for(int i = 2; i < (int)words.size(); ++i) {
          if( words[i] == "id"          ) iid = i - 2;
          if( words[i] == "x"           ) ix  = i - 2;
          if( words[i] == "y"           ) iy  = i - 2;
          if( words[i] == "z"           ) iz  = i - 2;
          if( words[i] == "c_orient[1]" ) iq1 = i - 2;
          if( words[i] == "c_orient[2]" ) iq2 = i - 2;
          if( words[i] == "c_orient[3]" ) iq3 = i - 2;
          if( words[i] == "c_orient[4]" ) iq4 = i - 2;
        }
        if( iid < 0 || ix < 0 || iy < 0 || iz < 0 )
          break;
        bQuat = (iq1 > -1 && iq2 > -1 && iq3 > -1 && iq4 > -1);
        for(int i = 0; i < nmol; ++i) {
          getline(fin, line);
          words = String(line).Split();
          mid.push_back(words[iid].ToInt() - 1);
          posx.push_back(words[ix].ToDouble());
          posy.push_back(words[iy].ToDouble());
          posz.push_back(words[iz].ToDouble());
          if( bQuat ) qua1.push_back(words[iq1].ToDouble());
          if( bQuat ) qua2.push_back(words[iq2].ToDouble());
          if( bQuat ) qua3.push_back(words[iq3].ToDouble());
          if( bQuat ) qua4.push_back(words[iq4].ToDouble());
        }
      }
      continue;
    }

    //# Make input for MALIO #
    vector<Vector3> coord(nmol);
    vector<Quaternion> direct(nmol);
    for(int i = 0; i < nmol; ++i) {
      coord[i].x = posx[i];
      coord[i].y = posy[i];
      coord[i].z = posz[i];
      direct[i].w   = qua1[i];
      direct[i].v.x = qua2[i];
      direct[i].v.y = qua3[i];
      direct[i].v.z = qua4[i];
    }
    Vector3 sim_box(lx, ly, lz);

    if( !bQuat ) {
      bool bST = false;
      for(int i = 0; i < (int)op_settings.analysis_type.size(); ++i) {
        if( op_settings.analysis_type[i] == "S" ) bST = true;
        if( op_settings.analysis_type[i] == "T" ) bST = true;
      }
      if( bST ) {
        cerr << "ERROR: Order Parameter S or T needs qurtanions in " << file_name
             << endl;
        exit(1);
      }
    }

    // # Use MALIO #
    DataFrame df = ml_lsa::op_analyze(coord, direct, sim_box, op_settings);

    const vector<string>& df_headers = df.GetHeader();
    const vector<vector<double>>& op = df.GetOrderParameter();

    vector<int> iheaders;
    for(int i = 0; i < (int)headers.size(); ++i) {
      for( int ih = 0; ih < (int)df_headers.size(); ++ih ) {
        if( df_headers[ih] == headers[i] ) {
          iheaders.push_back(ih);
          break;
        }
      }
    }
    
    string sFile = "__tmp/" + data_frame + filename_list_p[n] + ".csv";
    ofstream ofs(sFile.c_str());
    if( ofs.fail() ) {
      cerr << "Failed to open " << sFile << endl; 
      return 1;
    }
    ofs << fixed << setprecision(16);
    for( int imol = 0; imol < nmol; ++imol ) {
      ofs << imol;
      for( int i = 0; i < (int)iheaders.size(); ++i ) {
        int ih = iheaders[i];
        ofs << "," << op[imol][ih];
      }
      ofs << endl;
    }
    ofs.close();
  }

  // Write elapsed time #
  if( rank == 0 ) {
    clock_t t2 = clock();
    double time = double(t2 - t1) / CLOCKS_PER_SEC;
    cout << "### Record ### Elap.Time:" << time <<  "[sec]" << endl;
  }

  MPI_Finalize();
  
  return 0;
}

