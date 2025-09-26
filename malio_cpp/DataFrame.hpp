
#pragma once

#include <vector>
#include <map>
#include <string>

using namespace std;

class DataFrame
{
public:
  DataFrame(const map<string, vector<double>>& op, const string& l);
  DataFrame(){}
  const vector<string>& GetHeader() const { return header; }
  const vector<vector<double>>& GetOrderParameter() const { return order_parameter; }
  const string& GetLabel() const { return label; }
  void AddColumns(const DataFrame& df);
  void SetLabel(const string& l) { label = l; }
  static void SendToRoot(const DataFrame& df1, DataFrame& df2, int irank);

private:
  void SetHeader(const vector<string>& h) { header = h; }
  void SetOrderParameter(const vector<vector<double>>& op) { order_parameter = op; }
  
private:
  vector<string> header;
  vector<vector<double>> order_parameter;
  string label;
};


