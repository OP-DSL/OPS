#!/usr/bin/perl

use strict;
use warnings;
#my $file = './Tests/OMP4/nvprofCloverleafOMP4EventsAndMetrics.csv';

my $file =  $ARGV[0];

my $cudaFile = 0;

#print "file = $file ";


sub getNumberOfEventsorMetrics {
  my ($row) = @_;
  open my $infoMetrics, $file or die "Could not open $file: $!";


  my @metric = split /,/, $row;
  my $nameFirstKernel = $metric[1]; # kernel name;
  my $numberofmetricsReal = 0;
  #print " name to be checked $nameFirstKernel \n";
  my $l = <$infoMetrics>;
  while( $l = <$infoMetrics> ) {
    if( $l eq $row ) {
      last;
    }
  }

  my @output;

  my @metricTmp = split /,/, $l;
  #print "confront between $metricTmp[1] and $nameFirstKernel \n ";
  if(($metricTmp[1] eq $nameFirstKernel)){
    $numberofmetricsReal++;
    push @output,$metricTmp[3];
  }

  while( my $lineTmp = <$infoMetrics>)
  {
    #print "$lineTmp \n " ;
    my @metricTmp = split /,/, $lineTmp;
    #print "confront between $metricTmp[1] and $nameFirstKernel \n ";
    if(!($metricTmp[1] eq $nameFirstKernel))
    {
      last;
    }
    $numberofmetricsReal++;
    push @output,$metricTmp[3];
    #print "$numberofmetricsReal ";
  }

  unshift(@output,$numberofmetricsReal);
  return @output;
}




my $checkevent = 0;

open my $info2, $file or die "Could not open $file: $!";
while (my $l = <$info2>) {

  if ($l =~ /.*Event result:/g) {
    # print $l;
    $checkevent = 1;
  }
}


open my $info, $file or die "Could not open $file: $!";
my $line = <$info>;
my $numberofevents = 0;
my $numberofmetrics = 0;
my $cont = 0;
while($cont < 4)
{
  $line = <$info>;
  #print $line;
  if (index($line, "cuda") != -1) {
     $cudaFile = 1;
  } 
  $cont++;
}

#print "cudaFile = $cudaFile ";

#print $line;
my $foundMetrics = 0;
my $wroteEventsHead = 0;



while( my $line = <$info>)  {
  my $newrow;
  #print $line;


  if( $checkevent == 1 && $wroteEventsHead == 0){

    #print $line;
    $newrow = $line;
    chomp $newrow;
    #print "$newrow\n";
    $line = <$info>;
    $newrow = $line;
    #$line = <$info>;
    my @event = &getNumberOfEventsorMetrics($line);
    $numberofevents = $event[0];
    my $eventsoutput = "\"Device\", \"Kernel\",\"Invocations\"";
    my $eventsoutputSecondRow = ",,";
    $cont = 1;
    while($cont <= ($event[0]))
    {
      $eventsoutput = "$eventsoutput,\"$event[$cont]\",,";
      $eventsoutputSecondRow = "$eventsoutputSecondRow,\"Min\",\"Max\",\"Avg\"";
      $cont++;
    }
    print "Event result:\n";
    print "$eventsoutput\n";
    print "$eventsoutputSecondRow\n";
    $wroteEventsHead = 1;
  }

#print "checkevent = $checkevent  ----------- foundMetrics = $foundMetrics \n ";
  if(($checkevent == 1 && $line eq "\n") || ($checkevent == 0 && $foundMetrics ==0)){
    $newrow = $line;
    #chomp $newrow;
    $line = <$info>;
    $newrow = $line;
    if($checkevent == 1){
      $line = <$info>;
      $line = <$info>;
      $newrow = $line;
    }


    #print "$newrow\n";
    #I am splitting "Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg" in order to count the number of metrics that nvprof have got,
    # because those can be flexible.
    #input $newrow;

    my @metric = &getNumberOfEventsorMetrics($newrow);

   #print "\n \n metrics = @metric\n \n";
    $numberofmetrics = $metric[0];

    #  my @metric = split /,/, $metrics;
    my $metricsoutput = "\"Device\", \"Kernel\",\"Invocations\"";
    my $metricsoutputSecondRow = ",,";
    $cont = 1;
    while($cont <= ($metric[0]))
    {
      $metricsoutput = "$metricsoutput,$metric[$cont],,";
      $metricsoutputSecondRow = "$metricsoutputSecondRow,\"Min\",\"Max\",\"Avg\"";
      $cont++;
    }
    chomp $newrow;
    #print "$newrow\n";
    print "Metric result:\n";
    print "$metricsoutput\n";
    print "$metricsoutputSecondRow\n";
    $foundMetrics = 1;
  }
  #print "find = $foundMetrics\n";
  if($foundMetrics == 1){
    #found Metrics
    chomp $line;

    my @substr = split /,/, $line;
    #print $substr[6];
    my ($correctName);

    if($cudaFile == 1){
       ($correctName) = $substr[1] =~ /(?<=ops_)[a-zA-Z0-9_.-]*(?=_[a-zA-Z0-9]*)/g;
  
    }
    else
    {
         ($correctName) = $substr[1] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;

    }

    $correctName = "" unless $correctName;
    $correctName = "\"$correctName\"";


    #print "correct name $correctName \n";


    if ($correctName eq "\"\"" && $cudaFile){
      $newrow = $line;
      chomp $newrow;
    }else{

      $newrow = "$substr[0],$correctName,$substr[2],$substr[5],$substr[6],$substr[7]";

      my $cont1 = 0;
      while($cont1 < ($numberofmetrics-1)){

        $line = <$info>;
        chomp $line;
        my @substr2 = split /,/, $line;
        $newrow = "$newrow, $substr2[5],$substr2[6],$substr2[7]";
        $cont1++;
      }

      chomp $newrow;
    }

  }

  else{

    chomp $line;

    my @substr = split /,/, $line;
    #print $substr[6];
    my ($correctName);

    if($cudaFile == 1){
       ($correctName) = $substr[1] =~ /(?<=ops_)[a-zA-Z0-9_.-]*(?=_[a-zA-Z0-9]*)/g;
    }
    else
    {
         ($correctName) = $substr[1] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;
    }

    $correctName = "" unless $correctName;
    $correctName = "\"$correctName\"";

    


    if ($correctName eq "\"\""){
      $newrow = $line;
      chomp $newrow;
    }else{
      $newrow = "$substr[0],$correctName,$substr[2],$substr[4],$substr[5],$substr[6]";

      my $cont1 = 0;
      while($cont1 < ($numberofevents-1)){
        $line = <$info>;
        chomp $line;
        my @substr2 = split /,/, $line;
        $newrow = "$newrow, $substr2[4],$substr2[5],$substr2[6]";
        $cont1++;
      }

      chomp $newrow;
    }




  }


  print "$newrow\n";

}
