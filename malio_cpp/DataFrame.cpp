
#include <iostream>
#include <mpi.h>
#include "DataFrame.hpp"
#include "String.h"


DataFrame::DataFrame(const map<string, vector<double>>& op, const string& l)
  : label(l)
{
  int ncol = op.size();
  if( ncol == 0 )
    return;
  int nrow = op.begin()->second.size();
  
  header.resize(ncol);
  order_parameter.resize(nrow, vector<double>(ncol));

  int i = 0;
  for(auto col : op) {
    header[i] = col.first;
    for(int j = 0; j < nrow; ++j) {
      order_parameter[j][i] = col.second[j];
    }
    ++i;
  }
}

void DataFrame::AddColumns(const DataFrame& df)
{
  const vector<string>& h = df.GetHeader();
  const vector<vector<double>>& op = df.GetOrderParameter();

  int nrow = op.size();
  
  if( order_parameter.size() == 0 ) {
    order_parameter.resize(nrow);
  }
  
  header.insert(header.end(), h.begin(), h.end());
  for(int i = 0; i < nrow; ++i) {
    order_parameter[i].insert(order_parameter[i].end(), op[i].begin(), op[i].end());
  }

}


void DataFrame::SendToRoot(const DataFrame& df1, DataFrame& df2, int irank)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if( irank == 0 ) {
    if( rank == 0 ) {
      df2 = df1;
    }
    return;
  }

  if( rank == irank ) {
    string s = df1.GetLabel() + "\n";
    const vector<string>& h = df1.GetHeader();
    for(int i = 0; i < (int)h.size(); ++i) {
      s += h[i] + "\n";
    }
    int ns = s.size();
    MPI_Send(&ns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(s.c_str(), ns, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

    const vector<vector<double>>& order_parameter = df1.GetOrderParameter();
    int n = order_parameter.size();
    int m = order_parameter[0].size();
    MPI_Send(&n, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&m, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);

    vector<double> op(n*m);
    int k = 0;
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < m; ++j) {
        op[k] = order_parameter[i][j];
        ++k;
      }
    }
    MPI_Send(&op[0], n * m, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
  }
    
  if( rank == 0 ) {
    int ns;
    MPI_Status st;
    MPI_Recv(&ns, 1, MPI_INT, irank, 0, MPI_COMM_WORLD, &st);
    char c[ns+1];
    MPI_Recv(c, ns, MPI_CHAR, irank, 1, MPI_COMM_WORLD, &st);
    c[ns] = '\0';
    String s = c;;
    vector<String> vs = s.Split("\n");
    df2.SetLabel(vs[0]);
    vector<string> header;
    for(int i = 1; i < (int)vs.size(); ++i) {
      header.push_back(vs[i]);
    }
    df2.SetHeader(header);

    int n, m;
    MPI_Recv(&n, 1, MPI_INT, irank, 2, MPI_COMM_WORLD, &st);
    MPI_Recv(&m, 1, MPI_INT, irank, 3, MPI_COMM_WORLD, &st);

    vector<double> op(n * m);
    MPI_Recv(&op[0], n * m, MPI_DOUBLE, irank, 4, MPI_COMM_WORLD, &st);
    
    vector<vector<double>> order_parameter(n, vector<double>(m));
    int k = 0;
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < m; ++j) {
        order_parameter[i][j] = op[k];
        ++k;
      }
    }
    df2.SetOrderParameter(order_parameter);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}




