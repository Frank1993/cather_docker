#!/usr/bin/env python

import argparse

# main: where all the magic happens
if __name__ == "__main__":

    print("Testing process for submitting a Docker")
	
    # parse the command line
    parser = argparse.ArgumentParser(description='Test the dockerfile', formatter_class=argparse.RawDescriptionHelpFormatter)
    #parser.add_argument('-i', '--ids',      help='get only the ids of the users',                default=False, required=False,  action='store_true')
    #parser.add_argument('-p', '--print',    help='print the results of ".xssh"',                 default=False, required=False,  action='store_true')
    #parser.add_argument('-t', '--test',     help='only tests if user(s) are added',              default=False, required=False,  action='store_true')
    #parser.add_argument('-e', '--error',    help='do not print error messages',                  default=False, required=False,  action='store_true')
    #parser.add_argument('-s', '--serial',   help='run commands serial instead of parallel',      default=False, required=False,  action='store_true')
    #parser.add_argument('-r', '--reconfig', help='run /opt/bin/reconfigure on all machines',     default=False, required=False,  action='store_true')
    #parser.add_argument('-c', '--clusters', help='clusters to add user(s) (i.e. "ccp gcr ...")', default='cpp', required=False)
    #parser.add_argument('users', help='user(s) to add to the cluster(s) in the form: alias@domand', nargs='+')
    args = parser.parse_args()

    # print the args to the screen for debugging
    print(args)
    print("")