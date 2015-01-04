import sys
import time
from IPython import parallel as p
from IPython.display import clear_output



def wait_watching_stdout(ar, rc, truncate=1000):
    '''
    Method to print elapsed time of all processes on all engines to stdout.

    keyword args:
    ar --- list of engine jobs, created with ar.append(lview.apply_async(xxx))
    rc --- The ipython client
    truncate --- accuracy of timing.
    '''
    # keep track of which jobs have finished and init to false for all
    ready = []
    for i in xrange(len(ar)):
        ready.append(False)
    allReady = False
    # loop until all jobs are finished
    while not allReady:
        count = 0
        # loop through all of the engines
        for eng in ar:
            # flush
            rc.spin()
            # if it isn't ready, then check that it is running
            if not eng.ready():
                # get the stdout from this engine
                stdout = [ rc.metadata[msg_id]['stdout'] for msg_id in eng.msg_ids ]
                # not printing any stdout, leave for now
                if not any(stdout):
                    continue
                # clear_output doesn't work in plain terminal / script environments
                clear_output()
                print '-' * 30
                print "%.3fs elapsed" % eng.elapsed
                print ""
                # print the actual stdout
                for stdo in stdout: 
                    if stdo:
                        print "[ stdout ]\n%s" % (stdo[-truncate:])
                # flush the output stream
                sys.stdout.flush()
            # if the engine is ready, mark it
            else:
                ready[count] = True
            count += 1
        allReady = all(x == True for x in ready)
        # sleep for 2ms and check again
        time.sleep(2)
