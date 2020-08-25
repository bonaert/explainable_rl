for filename in run_without_*.pbs; do
    echo "Submitting job $filename then sleeping for 20 seconds"
    qsub "$filename"
    sleep 20
done
