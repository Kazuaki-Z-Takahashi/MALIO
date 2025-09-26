
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <numeric>
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
  int fnum = 0;
  string dirname = "structure";
  string data_frame = "data_frame_all.csv";
  int nstruct = 0;
  
  for(int i = 1; i < argc; ++i) {
    string s = argv[i];
    if( (s == "-os" || s == "--op_settings") && i + 1 < argc ) {
      os_file = argv[i+1];
      ++i;
      continue;
    }
    if( (s == "-fn" || s == "--number_of_files") && i + 1 < argc ) {
      fnum = String(argv[i+1]).ToIntDef(0);
      ++i;
      continue;
    }
    if( (s == "-dir" || s == "--directory_name") && i + 1 < argc ) {
      dirname = argv[i+1];
      ++i;
      continue;
    }
    if( (s == "-df" || s == "--data_frame_name")  && i + 1 < argc ) {
      data_frame = argv[i+1];
      ++i;
      continue;
    }
    if( (s == "-n" || s == "--number_of_structures")  && i + 1 < argc ) {
      nstruct = String(argv[i+1]).ToIntDef(0);
      ++i;
      continue;
    }
  }

  if( os_file == "" || fnum < 1 || data_frame == "" || nstruct < 1 ) {
    cerr << "Usage: " << argv[0] << " -os OP_SETTINGS_JSON "
         << "-fn NUMBER_OF_INPUT_FILES -dir DIRECTORY_NAME -df DATA_FRAME_NAME "
         << "-n NUMBER_OF_STRUCTURES"
         << endl;
    exit(1);
  }

  OP_SETTINGS op_settings;
  op_settings.ReadJson(os_file);

  vector<string> dirs(nstruct);
  for(int i = 0; i < nstruct; ++i) {
    dirs[i] = "__input/" + dirname + String(i+1) + "/";
  }
  
  vector<DataFrame> data_frames;
  
  for( int is = 0; is < nstruct; ++is ) { 
  
    //### Compute LOPs for "structure_1/2" ###
  
    //# Make list of file names to be readed #
    vector<string> filename_list;
    for(int i = 0; i < fnum; ++i) {
      ostringstream oss;
      oss << std::setw(7) << std::setfill('0') << i;
      string s = dirs[is] + oss.str() + ".txt";
      filename_list.push_back(s);
    }

    //# Loop for LOPs calculation start #

    clock_t t1 = clock();
    int n10 = 10;
    for(int n = rank; n < fnum; n += nproc) {
      const string& file_name = filename_list[n];
      
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
      int ix = -1, iy = -1, iz = -1, iq1 = -1, iq2 = -1, iq3 = -1, iq4 = -1;
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
            if( words[i] == "x"           ) ix  = i - 2;
            if( words[i] == "y"           ) iy  = i - 2;
            if( words[i] == "z"           ) iz  = i - 2;
            if( words[i] == "c_orient[1]" ) iq1 = i - 2;
            if( words[i] == "c_orient[2]" ) iq2 = i - 2;
            if( words[i] == "c_orient[3]" ) iq3 = i - 2;
            if( words[i] == "c_orient[4]" ) iq4 = i - 2;
          }
          if( ix < 0 || iy < 0 || iz < 0 )
            break;
          bQuat = (iq1 > -1 && iq2 > -1 && iq3 > -1 && iq4 > -1);
          for(int i = 0; i < nmol; ++i) {
            getline(fin, line);
            words = String(line).Split();
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

      /*
      cout << nmol << endl;
      cout << lx << " " << ly << " " << lz << endl;
      cout << posx[0] << endl;
      cout << posx[nmol-1] << endl;
      cout << qua1[0] << endl;
      cout << qua1[nmol-1] << endl;
      */

      if( posx.empty() ) {
        cerr << "ERROR: Coordinates not found in " << file_name
             << endl;
        exit(1);
      }

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
      
      //# Make input for ML-LSA #
      vector<Vector3> coord(nmol);
      vector<Quaternion> direct(nmol);
      for(int i = 0; i < nmol; ++i) {
        coord[i].x = posx[i];
        coord[i].y = posy[i];
        coord[i].z = posz[i];
        if( bQuat ) {
          direct[i].w   = qua1[i];
          direct[i].v.x = qua2[i];
          direct[i].v.y = qua3[i];
          direct[i].v.z = qua4[i];
        }
        else {
          direct[i].w   = 1.0;
          direct[i].v.x = 0.0;
          direct[i].v.y = 0.0;
          direct[i].v.z = 0.0;          
        }
      }
      Vector3 sim_box(lx, ly, lz);

      DataFrame df = ml_lsa::op_analyze(coord, direct, sim_box, op_settings);
      df.SetLabel("structure_" + to_string(is+1));

      data_frames.push_back(df);
    }

    // Write elapsed time #
    if( rank == 0 ) {
      clock_t t2 = clock();
      double time = double(t2 - t1) / CLOCKS_PER_SEC;
      cout << "### Record ### Elap.Time:" << time <<  "[sec]" << endl;
    }
  }

  //### Marge data frames of LOPs ###

  vector<DataFrame> data_frames_all;
  int i = 0;
  for( int is = 0; is < nstruct; ++is ) { 
    int irank = 0;
    for(int n = 0; n < fnum; ++n) {
      DataFrame df1, df2;
      if( irank == rank )
        df1 = data_frames[i];
      DataFrame::SendToRoot(df1, df2, irank);
      if( rank == 0 ) {
        data_frames_all.push_back(df2);
      }
      if( rank == irank ) {
        ++i;
      }
      ++irank;
      irank = irank % nproc;
    }
  }

  if( rank == 0 ) {
    path tmpdir = "__tmp";
    if (!exists(tmpdir)) {
      create_directory(tmpdir);
    }
    misc::write_csv(data_frames_all, "__tmp/" + data_frame);
  }
  
  MPI_Finalize();
  
  return 0;
}

