/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief OPS exceptions
  * @author Istvan Reguly
  * @details declarations for throwing and querying exceptions in OPS
  */

#ifndef __OPS_EXCEPTIONS_H
#define __OPS_EXCEPTIONS_H

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstring>

#define OPS_NOT_IMPLEMENTED 1
#define OPS_RUNTIME_CONFIGURATION_ERROR 2
#define OPS_INTERNAL_ERROR 3
#define OPS_RUNTIME_ERROR 4
#define OPS_HDF5_ERROR 5
#define OPS_INVALID_ARGUMENT 6
#define OPS_OPENCL_ERROR 7
#define OPS_OPENCL_BUILD_ERROR 7

struct OPSException : public std::exception
{
    int code;
    const static int maxsize = 2048;
    unsigned char _data[maxsize];

    int cursize;
    int ridx;
    std::stringstream msg;

    /* The copy of the underlying string returned by stringstream::str() is a temporary 
     * object that will be destructed at the end of the expression, so directly calling 
     * c_str() on the result of str() (for example in auto *ptr = out.str().c_str();) 
     * results in a dangling pointer.  Hence we need to get the message out of 'msg'
     * and stick it in heap memory somewhere before we return it via what(), otherwise
     * programs that catch the exception will get a dangling pointer which may or may 
     * not point at a valid string.  I caught this via gtest, where I got a garbage
     * message.
     */
    mutable std::string persistentMsg;

    virtual ~OPSException() throw() {}
    OPSException(int code) : code(code), cursize(0), ridx(0) { }
    OPSException(const OPSException &ex2) {
      code = ex2.code;
      memcpy(_data, ex2._data, maxsize*sizeof(unsigned char));
      cursize = ex2.cursize;
      ridx = ex2.ridx;
      msg << ex2.msg.rdbuf();
    }

    template<class T>
    OPSException& operator<< (const T& val)
    {  
       insert(val);
       return *this;
    }

    OPSException(int code, const char *val) : code(code), cursize(0), ridx(0) { *this << val; }

    void insert(const char *val)
    {  
       msg << val;
    }

    template<class T> 
    void insert(const T& val)
    {  
       if(sizeof(T) + cursize > maxsize )
       {  
          std::cerr << "Too many data items!\n";
          abort();
       }

       *(T*)(_data + cursize) = val;
       cursize += sizeof(T);

       msg << val;
    }

    virtual const char* what() const throw() {
       persistentMsg = msg.str();
       return persistentMsg.c_str();
    }

    template<class T>
       const T& data() {
          if(ridx + sizeof(T) > cursize) {
             std::cerr << "Reading too many data items!\n";
             abort();
          }
          const T& ret = *(T*)(_data + ridx);
          ridx += sizeof(T);
          return ret;
       }
 };



#endif /*__OPS_EXCEPTIONS_H*/
