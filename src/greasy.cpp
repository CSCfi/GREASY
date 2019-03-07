/*
 * This file is part of GREASY software package
 * Copyright (C) by the BSC-Support Team, see www.bsc.es
 *
 * GREASY is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * GREASY is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GREASY. If not, see <http://www.gnu.org/licenses/>.
 *
*/


#include "greasylog.h"
#include "greasyconfig.h"
#include "abstractengine.h"
#include "config.h"
#include "greasyutils.h"

#include <string>
#include <csignal>
#include <cstdlib>

#ifndef SYSTEM_CFG
#define SYSTEM_CFG "../etc/greasy.conf"
#endif
#ifndef USER_CFG
#define USER_CFG "./greasy.conf"
#endif

using namespace std;

AbstractEngine* engine = NULL;
int my_pid;
bool readConfig();
void termHandler( int sig );
void printError();

int main(int argc, char *argv[]) {
  my_pid=getpid();
  GreasyLog* log = GreasyLog::getInstance();
  GreasyConfig* config = GreasyConfig::getInstance();

  int opt;
  string input = "";
  // string filename = "";
  int ntasks_per_worker = 1;

  // check if there is a (non-option) argument:
  if ((argc <= 1) || (argv[argc - 1] == NULL) || (argv[argc - 1][0] == '-')) {
    printError();
    return 0;
  } else {
    input = argv[argc - 1];
  }

  // opterr = 0;
  // // Retrieve the options:
  // while ((opt = getopt(argc, argv, "n:f:")) != -1) {
  // // while ((opt = getopt(argc, argv, "f:")) != -1) {
  //   switch (opt) {
  //   case 'n':
  //     ntasks_per_worker = atoi(string(optarg).c_str());
  //     break;
  //   case 'f':
  //     filename = string(optarg);
  //     break;
  //   case '?':
  //     cerr << "? case\n";
  //     if (optopt == 'n') {
  //       cerr << "Option -n requires an argument." << endl;
  //     } else if (optopt == 'f') {
  //       cerr << "Option -f requires an argument." << endl;
  //     }
  //     cerr << "Unknown option: '" << char(optopt) << "'!" << endl;
  //     break;
  //   }
  // }

  // if (filename.length() < 1) {
  //   printError();
  //   return (0);
  // }

  if (argc != 2) {
      cout << "Usage: greasy filename" << endl;
      return (0);
  }

  string filename(argv[1]);

  // Read config
  if (!readConfig()) {
    cerr << "Failed to load config file" << endl;
    return -1;
  }

  if(config->keyExists("NTASKS_PER_WORKER")) {
    fromString(ntasks_per_worker, config->getValue("NTASKS_PER_WORKER"));
  }

  // Log Init
  if(config->keyExists("LogFile"))
    log->logToFile(config->getValue("LogFile"));

  if(config->keyExists("LogLevel")) {
    int logLevel = fromString(logLevel,config->getValue("LogLevel"));
    log->setLogLevel((GreasyLog::LogLevels)logLevel);
  }

  // Handle interrupting signals appropiately.
  signal(SIGTERM, termHandler);
  signal(SIGINT,  termHandler);
  signal(SIGUSR1, termHandler);
  signal(SIGUSR2, termHandler);

  // Create the proper engine selected and run it!
  engine = AbstractEngineFactory::getAbstractEngineInstance(filename,
    config->getValue("Engine"), ntasks_per_worker);
  if (!engine) {
      log->record(GreasyLog::error,"Greasy could not load the engine");
      return -1;
  }
  // Initialize the engine
  engine->init();

  // Start it. All tasks will be scheduled and run from now on
//   engine->dumpTasks();
  engine->run();

  // Finalize the engine once all tasks have finished.
  engine->finalize();

  // Ok we're done
  log->logClose();

}


bool readConfig () {

  GreasyConfig* config = GreasyConfig::getInstance();

  // First, try to read USER config. If it is not present, then use the System defaults.
  if (!config->readConfig(USER_CFG)) return config->readConfig(SYSTEM_CFG);

  return true;

}

void termHandler( int sig ) {
  char killTree[100];

  GreasyLog* log = GreasyLog::getInstance();
  log->record(GreasyLog::error, "Caught TERM signal");
  if (engine) engine->writeRestartFile();
  log->record(GreasyLog::error, "Greasy was interrupted. Check restart & log files");
  log->logClose();
  sprintf(killTree, "kill  -- -%d", my_pid);
  system(killTree);
  exit(1);
}


void printError() {
  cout << "Usage: greasy [OPTIONS] filename" << endl;
  cout << "  where [OPTIONS] can be:" << endl << endl;
  cout << " -n NTASKS\t\t Sets the number of tasks per worker to NTASKS" << endl;
  cout << " -f FILE\t\t Sets the input file to to FILE" << endl;
}
