
#include "String.h"

#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>

using namespace std;

String::String( int n )
{
  stringstream ss;
  ss << n;
  *this = ss.str();
}

String::String( double d )
{
  stringstream ss;
  ss << scientific;
  ss << setprecision(6);
  ss << setw(14);
  ss << uppercase;
  ss << right;
  ss << d;
  *this = ss.str();
}

vector<String> String::Split( const String& delimiter,
  bool ignore_blank ) const
{
  vector<String> vstr;
  SIZE_TYPE start = 0;
  SIZE_TYPE pos( find_first_of( delimiter ) );
  while( pos != NPOS ) {
    vstr.push_back( substr(start, pos-start) );
    start = pos + 1;
    pos = find_first_of( delimiter, start );
  }
  vstr.push_back( substr(start, length()-start) );
    
  if( ! ignore_blank ) {
    return vstr;
  }

  vector<String> vstr_;
  for( unsigned int i = 0; i < vstr.size(); ++i ) {
    if( vstr[i] != "" ) {
      vstr_.push_back( vstr[i] );
    }
  }
  return vstr_;
}


double String::ToDouble() const
{
  double d;
  stringstream ss( *this );
  ss >> d;
  if( ss.fail() ) throw "Failed to convert string to double";
  return d;
}

int String::ToInt() const
{
  int n;
  stringstream ss( *this );
  ss >> n;
  if( ss.fail() ) throw "Failed to convert stringe to int";
  return n;
}

int String::ToIntDef(int nDef) const
{
  int n;
  stringstream ss( *this );
  ss >> n;
  if( ss.fail() ) {
    return nDef;
  }
  else {
    return n;
  }
}

String String::ToUpper() const
{
  String ret = *this;
  transform( ret.begin(), ret.end(), ret.begin(), ::toupper );
  return ret;
}

String String::ToLower() const
{
  String ret = *this;
  transform( ret.begin(), ret.end(), ret.begin(), ::tolower );
  return ret;
}

String String::Trim() const
{
  SIZE_TYPE b = find_first_not_of(" \r\n");
  if( b == NPOS ) {
    return "";
  }
  SIZE_TYPE e = find_last_not_of(" \r\n");
  return substr( b, e - b + 1 );
}

String String::ReplaceAll( const String& key1,
  const String& key2 ) const
{
  String ret = *this;
  SIZE_TYPE pos( ret.find( key1 ) );
  while( pos != NPOS ) {
    ret.replace( pos, key1.length(), key2 );
    SIZE_TYPE start = pos + key2.length();
    pos = ret.find( key1, start );
  }
  return ret;
}

String String::SPrintf( const char* format, ... )
{
  int nBuf = 256;
  vector<char> buf(nBuf);
  while(1) {
    va_list argp;
    va_start(argp, format);
    int nSize = vsnprintf(&(buf[0]), nBuf, format, argp);
    if( 0 <= nSize && nSize < nBuf ) {
      break;
    }
    nBuf *= 2;
    buf.resize( nBuf );
    va_end(argp);
  }
  return String( &(buf[0]) );
}

