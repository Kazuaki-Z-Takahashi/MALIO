
#ifndef _STRING_H_
#define _STRING_H_

#include <string>
#include <vector>

using namespace std;

typedef string::size_type SIZE_TYPE;
const SIZE_TYPE NPOS = string::npos;

// derived class from std::string with useful functions 
class String : public string
{
public:
  String() {}
  String( const string& str ) : string(str) {}
  String( const char* p ) : string(p) {}
  String( int n );
  String( double d );
  ~String() {}

  String& operator=( const char* p )
  {
      *this = String( p );
      return *this;
  }
  String substr( SIZE_TYPE index, SIZE_TYPE length = NPOS ) const
  {
    return String( string::substr( index, length ) );
  }
  
  char& head() { return *begin(); };
  char& back() { return *rbegin(); };

  vector<String> Split( const String& delimiter = " \n\r\t",
    bool ignore_blank = true ) const;
  double ToDouble() const ;
  int ToInt() const;
  int ToIntDef(int nDef) const;
  String ToUpper() const;
  String ToLower() const;
  String Trim() const;
  String ReplaceAll( const String& key1, const String& key2 ) const;

  static String SPrintf( const char* format, ... );
};

#endif //_STRING_H_


