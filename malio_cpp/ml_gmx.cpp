
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

  if( os_file == "" || data_frame == "" || nstruct < 1) {
    cerr << "Usage: " << argv[0] << " -os OP_SETTINGS_JSON -dir DIRECTORY_NAME "
         << "-df DATA_FRAME_NAME -n NUMBER_OF_STRUCTURES"
         << endl;
    exit(1);
  }

  OP_SETTINGS op_settings;
  op_settings.ReadJson(os_file);

  
  const string fname = "__input/" + dirname;
  
  vector<DataFrame> data_frames;
  vector<string> filename_lists[nstruct];
  
  for( int is = 0; is < nstruct; ++is ) { 
  
    //### Compute OPs for "structure 1/2 (s1/s2)" trajectory . ###
  
    //# Make list of file names to be readed #
    vector<string> filename_list;
    for(int i = 0; ; ++i) {
      ostringstream oss;
      oss << is + 1 << "/" << std::setw(4) << std::setfill('0') << i + 1;
      string s = fname + oss.str() + ".txt";
      if( !filesystem::is_regular_file(s) )
	break;
      filename_list.push_back(s);
    }
    filename_lists[is] = filename_list;
    int nfile = filename_lists[0].size();
    if( nfile != (int)filename_list.size() ) {
      cerr << "ERROR: Number of files in " << fname << is + 1
           << " not match which in " << fname << 1 << endl;
      exit(1);
    }

    //# Loop for LOPs calculation start #

    clock_t t1 = clock();
    int n10 = 10;
    for(int n = rank; n < (int)filename_list.size(); n += nproc) {
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
        exit(1);
      }

      Vector3 sim_box;
      int nmol = 0;
      vector<Vector3> coord;
      vector<Vector3> direct;
      try {
        string line;
        getline(fin, line);
        
        vector<String> words = String(line).Split();
        sim_box.x = words[0].ToDouble();
        sim_box.y = words[1].ToDouble();
        sim_box.z = words[2].ToDouble();
      
        getline(fin, line);
        nmol = stoi(line);
      
        coord.resize(nmol);
        direct.resize(nmol);
        for(int i = 0; i < nmol; ++i) {
          getline(fin, line);
          words = String(line).Split();

          Vector3 v;
          v.x = words[0].ToDouble();
          v.y = words[1].ToDouble();
          v.z = words[2].ToDouble();
          coord[i] = v;
        
          v.x = words[3].ToDouble();
          v.y = words[4].ToDouble();
          v.z = words[5].ToDouble();
          direct[i] = v;
        }
      }
      catch(...) {
        fin.close();
        cerr << "ERROR: Failed to read " << file_name << endl;
        exit(1);
      }
      fin.close();

      DataFrame df = ml_lsa::op_analyze(coord, direct, sim_box, op_settings);
      df.SetLabel("s" + to_string(is+1));

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
    for(int n = 0; n < (int)filename_lists[is].size(); ++n) {
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

