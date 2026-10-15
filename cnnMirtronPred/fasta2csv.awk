#!/usr/bin/awk -f
{
  SEP=","
  if (substr($1,1,1)==">")
    if (NR>1)
      printf "\n%s%s", substr($0,2,length($0)-1), SEP
    else 
      printf "%s%s", substr($0,2,length($0)-1), SEP
  else
    printf "%s", $0
}END{printf "\n"}