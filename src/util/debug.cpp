#include "debug.h"
#include <mutex>

//#define ENABLE_DEBUG_LOG

namespace debug
{
	struct debug_log_t
	{
		alignas(64) std::mutex mtx;
		umap<std::string,std::vector<std::string>> msg_by_cat = {};
		i32 max_size = 1000;
	};

	debug_log_t* dl = nullptr;

	bool start_logger()
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl != nullptr)
		{
			printf("Debug log already started!\n");
			return false;
		}

		dl = new debug_log_t();
		return true;
#else
		return false;
#endif
	};

	bool stop_logger()
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Debug log not running!\n");
			return false;
		}

		delete dl;
		dl = nullptr;
		return true;
#else
		return false;
#endif		
	};

	void log(std::string category, std::string message)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to log message. Debug log not started!\n");
			return;
		}

		{
			std::unique_lock lock(dl->mtx);
			if(dl->msg_by_cat.find(category) == dl->msg_by_cat.end())
				dl->msg_by_cat.insert({category,{}});
			dl->msg_by_cat.at(category).push_back(message);
			if(dl->msg_by_cat.at(category).size() > dl->max_size)
				dl->msg_by_cat.erase(dl->msg_by_cat.at(category).front());
		}
#endif		
	};

	void log(std::string category, std::string label, math::array<float> arr)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to log message. Debug log not started!\n");
			return;
		}

		
		std::string array_string = "";
		std::string as = "";
		as.reserve(32);
		for(i32 i = 0; i < arr.size(); i++)
		{
			i32 len = sprintf(as.data(),"%5.3f",arr.at(i));
			for(i32 j = 0; j < len; j++)
				array_string.push_back(as.data()[j]);
			array_string += " ";
		}

		{
			std::unique_lock lock(dl->mtx);
			if(dl->msg_by_cat.find(category) == dl->msg_by_cat.end())
				dl->msg_by_cat.insert({category,{}});
			dl->msg_by_cat.at(category).push_back(label+" :: "+array_string);
			if(dl->msg_by_cat.at(category).size() > dl->max_size)
				dl->msg_by_cat.erase(dl->msg_by_cat.at(category).front());
		}
#endif		
	};

	void print_logs(std::string category)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to print logs. Debug log not started!\n");
			return;
		}

		{
			std::unique_lock lock(dl->mtx);
			if(dl->msg_by_cat.find(category) == dl->msg_by_cat.end())
			{
				printf("No messages for category: %s\n",category.c_str());
				return;
			}

			printf("Messages for category: %s\n",category.c_str());
			auto msgs = dl->msg_by_cat.at(category);
			for(i32 i = 0; i < msgs.size(); i++)
				printf("\t[%2i] %s\n",i,msgs.at(i).c_str());
		}
#endif		
	};

	void clear_logs(std::string category)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to clear logs. Debug log not started!\n");
			return;
		}

		{
			std::unique_lock lock(dl->mtx);
			if(dl->msg_by_cat.find(category) == dl->msg_by_cat.end())
				return;

			dl->msg_by_cat.at(category).clear();
		}
#endif		
	};

	i32 get_max_log_size()
	{
		i32 size = -1;
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to get max log size. Debug log not started!\n");
			return -1;
		}

		{
			std::unique_lock lock(dl->mtx);
			size = dl->max_size;
		}

#endif
		return size;
	};

	void set_max_log_size(i32 size)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to set max log size. Debug log not started!\n");
			return;
		}

		{
			std::unique_lock lock(dl->mtx);
			dl->max_size = size;
		}
#endif
	};

	void write_to_file(std::string category, std::string path, bool overwrite)
	{
#ifdef ENABLE_DEBUG_LOG
		if(dl == nullptr)
		{
			printf("Failed to write log file. Debug log not started!\n");
			return;
		}

		{
			std::unique_lock lock(dl->mtx);

			FILE* f = nullptr;
			if(overwrite)
				f = fopen(path.c_str(),"w+");
			else
				f = fopen(path.c_str(),"r+");

			if(f == nullptr)
			{
				printf("Failed to open log file for writing!\n");
				return;
			}
			
			if(dl->msg_by_cat.find(category) == dl->msg_by_cat.end())
			{
				printf("No messages for category: %s\n",category.c_str());
				return;
			}

			std::string pre = "Messages for category: " + category + "\n";
			fwrite(pre.c_str(),sizeof(char),pre.size(),f);

			auto msgs = dl->msg_by_cat.at(category);
			for(i32 i = 0; i < msgs.size(); i++)
			{
				std::string si = str(i);
				while(si.size() < 5)
					si.insert(si.begin(),' ');

				std::string s = "[" + si + "] " + msgs.at(i) + "\n";
				fwrite(s.c_str(),sizeof(char),s.size(),f);
			}

			fclose(f);
			f = nullptr;
		}
#endif
	};
};