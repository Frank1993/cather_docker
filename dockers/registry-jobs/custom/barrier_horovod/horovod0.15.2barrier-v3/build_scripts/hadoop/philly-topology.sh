#!/bin/bash

# Innvocation of this script is:
# bash philly-topology.sh <host1> <host2> etc
# Script is responsible for parsing philly-topology.txt file
# And returning rack info for each host that was passed as
# an argument.  ENSURE the file is executable.

# Supply appropriate rack prefix
RACK_PREFIX=default

# To test, supply a hostname as script input:
if [ $# -gt 0 ]; then

   CTL_FILE=${CTL_FILE:-"philly-topology.txt"}

   HADOOP_CONF=${HADOOP_CONF:-"/usr/local/hadoop/etc/hadoop"} 

   if [ ! -f ${HADOOP_CONF}/${CTL_FILE} ]; then
      echo -n "/$RACK_PREFIX/rack "
      exit 0
   fi

   # Iterate over each input arg
   while [ $# -gt 0 ] ; do
      nodeArg=$1
      exec< ${HADOOP_CONF}/${CTL_FILE}
      result=""
      # Iterate over each line in the topology file
      # Where host is found, add to results string
      while read line ; do
         ar=( $line )
         if [ "${ar[0]}" = "$nodeArg" ] ; then
            result="${ar[1]}"
         fi
      done
      shift
      if [ -z "$result" ] ; then
         echo -n "/$RACK_PREFIX/rack "
      else
         echo -n "$result "
      fi
   done

else
   echo -n "/$RACK_PREFIX/rack "
fi
