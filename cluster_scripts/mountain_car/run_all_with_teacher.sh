for filename in *with_teacher*.pbs; do
    echo "Submitting job $filename then sleeping for 20 seconds"
    qsub "$filename"
    sleep 20
done
