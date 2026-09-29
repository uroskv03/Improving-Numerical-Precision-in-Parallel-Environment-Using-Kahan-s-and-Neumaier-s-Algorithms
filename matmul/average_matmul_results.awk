BEGIN {
    FS = "|"
    OFS = " | "
    mode = ARGV[1]
    delete ARGV[1]
}

/^[[:space:]]*(Serial|Parallel)/ {
    algorithm = $1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", algorithm)

    if (mode == "serial" && algorithm !~ /^Serial/) next
    if (mode == "parallel" && algorithm !~ /^Parallel/) next

    for (column = 2; column <= 5; column++) {
        value = $column
        gsub(/[[:space:]]+/, "", value)
        sum[algorithm, column] += value
    }
    value = $6
    gsub(/[[:space:]]+/, "", value)
    sum[algorithm, 6] += value
    if (!(algorithm in minimum) || value < minimum[algorithm]) minimum[algorithm] = value
    if (!(algorithm in maximum) || value > maximum[algorithm]) maximum[algorithm] = value

    value = $7
    gsub(/[[:space:]]+/, "", value)
    sum[algorithm, 7] += value
    count[algorithm]++
    order[++algorithm_count] = algorithm
}

END {
    printf "%-18s | %-18s | %-18s | %-18s | %-18s | %-18s | %-8s\n",
            "Algoritam", "AbsErr(Double)", "RelErr(Double)", "AbsErr(Quad)",
            "RelErr(Quad)", "MaxDiff Avg", "MaxDiff Max", "MaxDiff Min", "Vreme Avg"
        printf "------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+----------\n"

    previous = ""
    for (position = 1; position <= algorithm_count; position++) {
        algorithm = order[position]
        if (seen[algorithm]) continue
        seen[algorithm] = 1
        previous = algorithm

         printf "%-18s | %-18e | %-18e | %-18e | %-18e | %-18e | %-18e | %-18e | %-8.4fs\n",
               algorithm,
               sum[algorithm, 2] / count[algorithm],
               sum[algorithm, 3] / count[algorithm],
               sum[algorithm, 4] / count[algorithm],
               sum[algorithm, 5] / count[algorithm],
               sum[algorithm, 6] / count[algorithm],
             maximum[algorithm],
             minimum[algorithm],
               sum[algorithm, 7] / count[algorithm]
    }
}