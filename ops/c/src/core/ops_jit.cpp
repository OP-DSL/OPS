#include <ops_lib_core.h>
#include <vector>
char *ops_generate_filename(const char *tag);
#include <dlfcn.h>
#ifdef OPS_JSON
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include <rapidjson/writer.h>
#include "rapidjson/prettywriter.h"
#include <fstream>


class JSONWriter {
protected:
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& allocator;

    JSONWriter() : allocator(d.GetAllocator()) {
         { d.SetObject(); }
    }
    rapidjson::Value stringToObject(const std::string& in) {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.SetString(in.c_str(), in.length(), allocator);
        return obj;
    }

public:
    void writeFile(const std::string& fileName) const {
      rapidjson::StringBuffer buffer;
      rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
      // writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
      writer.SetMaxDecimalPlaces(4);
      d.Accept(writer);
      std::ofstream outFile;
      outFile.open(fileName.c_str());
      outFile << buffer.GetString();
      outFile.close();
    }
};

/*namespace parser {
    struct JITData : public jsond::JSONDecodable<JITData> {
        struct Dat : public jsond::JSONDecodable<Dat> {

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, idx);
            DECODABLE_MEMBER(int, dim);
            DECODABLE_MEMBER(std::vector<int>, size);
            DECODABLE_MEMBER(std::vector<int>, base);
            DECODABLE_MEMBER(std::vector<int>, d_m);
            DECODABLE_MEMBER(std::vector<int>, d_p);
            DECODABLE_MEMBER(std::string, name);
            DECODABLE_MEMBER(std::string, type);
            END_MEMBER_DECLARATIONS();
        };
        struct Kernel : public jsond::JSONDecodable<Kernel> {
            struct Arg : public jsond::JSONDecodable<Arg> {
                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(int, argtype);
                DECODABLE_MEMBER(int, acc);
                DECODABLE_MEMBER(int, datidx);
                END_MEMBER_DECLARATIONS();
            };
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(int, index);
            DECODABLE_MEMBER(int, dim);
            DECODABLE_MEMBER(std::vector<int>, range);
            DECODABLE_MEMBER(std::vector<Arg>, args);
            DECODABLE_MEMBER(std::string, name);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Dat>, dats);
        DECODABLE_MEMBER(std::vector<Kernel>, kernels);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser
*/

class OPSKernelWriter : public JSONWriter {
  public:
    explicit OPSKernelWriter(OPS_instance *instance, const std::vector<ops_kernel_descriptor *>& ops_kernel_list) {
        rapidjson::Value dats(rapidjson::kArrayType);
        ops_dat_entry *item;
        TAILQ_FOREACH(item, &instance->OPS_dat_list, entries) {
            ops_dat dat = item->dat;
            rapidjson::Value currentDat(rapidjson::kObjectType);
            currentDat.AddMember("idx", dat->index, allocator);
            currentDat.AddMember("dim", dat->dim, allocator);
            rapidjson::Value size(rapidjson::kArrayType);
            rapidjson::Value base(rapidjson::kArrayType);
            rapidjson::Value d_m(rapidjson::kArrayType);
            rapidjson::Value d_p(rapidjson::kArrayType);
            for (int i = 0; i < dat->block->dims; i++) {
              size.PushBack(dat->size[i], allocator);
              base.PushBack(dat->base[i], allocator);
              d_m.PushBack(dat->d_m[i], allocator);
              d_p.PushBack(dat->d_p[i], allocator);
            }
            currentDat.AddMember("size", size, allocator);
            currentDat.AddMember("base", base, allocator);
            currentDat.AddMember("d_m", d_m, allocator);
            currentDat.AddMember("d_p", d_p, allocator);
            /*currentDat.AddMember("size", std::vector<int>(dat->size,&dat->size[dat->dim]), allocator);
            currentDat.AddMember("base", std::vector<int>(dat->base,&dat->base[dat->dim]), allocator);
            currentDat.AddMember("d_m", std::vector<int>(dat->d_m,&dat->d_m[dat->dim]), allocator);
            currentDat.AddMember("d_p", std::vector<int>(dat->d_p,&dat->d_p[dat->dim]), allocator);*/
            rapidjson::Value name(dat->name, allocator);
            currentDat.AddMember("name", name, allocator);
            rapidjson::Value type(dat->type, allocator);
            currentDat.AddMember("type", type, allocator);
            dats.PushBack(currentDat, allocator);
        }
        d.AddMember("dats", dats, allocator);


        rapidjson::Value kernels(rapidjson::kArrayType);
        for (unsigned idx = 0; idx < ops_kernel_list.size(); idx++) {
            ops_kernel_descriptor *kernel = ops_kernel_list[idx];
            rapidjson::Value currentKernel(rapidjson::kObjectType);
            currentKernel.AddMember("idx", kernel->index, allocator);
            currentKernel.AddMember("dim", kernel->dim, allocator);
            rapidjson::Value name(kernel->name, allocator);
            currentKernel.AddMember("name", name, allocator);
            rapidjson::Value range(rapidjson::kArrayType);
            for (int i = 0; i < 2*kernel->dim; i++)
              range.PushBack(kernel->range[i], allocator);
            currentKernel.AddMember("range", range, allocator);
            rapidjson::Value args(rapidjson::kArrayType);
            for (int arg = 0; arg < kernel->nargs; arg++) {
                rapidjson::Value currentArg(rapidjson::kObjectType);
                currentArg.AddMember("acc", kernel->args[arg].acc,allocator);
                currentArg.AddMember("argtype", kernel->args[arg].argtype,allocator);
                int datidx = -1;
                if (kernel->args[arg].argtype == OPS_ARG_DAT) 
                    datidx = kernel->args[arg].dat->index;
                currentArg.AddMember("datidx", datidx,allocator);
                args.PushBack(currentArg, allocator);
            }
            currentKernel.AddMember("args", args, allocator);
            kernels.PushBack(currentKernel, allocator);
        }
        d.AddMember("kernels", kernels, allocator);
    }
};

void ops_jit_write_json(OPS_instance *instance, const std::vector<ops_kernel_descriptor *>& ops_kernel_list, const char* tag) {
    OPSKernelWriter writer(instance, ops_kernel_list);
    char *filename= ops_generate_filename(tag);
    strcat(filename, "json");
    writer.writeFile(filename);
    ops_free(filename);
}
#else
void ops_jit_write_json(OPS_instance *instance, const std::vector<ops_kernel_descriptor *>& ops_kernel_list, const char *tag) {
    (void*)instance;
}
#endif

char *ops_generate_filename(const char *tag) {
  const char *tmpdir = getenv("TMPDIR");
  char *filename = (char*)ops_malloc((tmpdir ? strlen(tmpdir) : 4)+100+strlen(tag));
  memset(filename,0,(tmpdir ? strlen(tmpdir) : 4)+100+strlen(tag));
  if (tmpdir)
    strcat(filename,tmpdir);
  else
    strcat(filename,"/tmp");
  strcat(filename,"/opsjit_");
  strcat(filename,tag);
  strcat(filename,".");
  return filename;
}

void ops_jit_compile(const char * tag) {
  char *filename = ops_generate_filename(tag);
  strcat(filename, "so");
  char *command = (char*)ops_malloc(strlen(filename)+100);
  command[0] = '\0';
  strcat(command, "make ");
  strcat(command, filename);
  ops_printf("Calling %s\n", command);
  int retval = system(command);
  if (retval != 0) throw OPSException(OPS_INTERNAL_ERROR,"JIT Command failed");
}

void ops_jit_run(const char *tag) {
  char *filename = ops_generate_filename(tag);
  strcat(filename, "so");
  ops_printf("Opening %s\n", filename);
  void *handle = dlopen(filename, RTLD_LAZY);
  ops_printf("error %s\n", dlerror());
  if (handle == nullptr) throw OPSException(OPS_INTERNAL_ERROR,dlerror());
  void (*func_print_name)(const char*);
  *(void**)(&func_print_name) = dlsym(handle, "print_name");
  if (func_print_name) {
    func_print_name("testing");
  } else throw OPSException(OPS_INTERNAL_ERROR,"JIT function not found in shared object");
}

