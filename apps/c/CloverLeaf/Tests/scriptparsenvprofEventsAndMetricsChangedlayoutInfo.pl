#!/usr/bin/perl

use strict;
use warnings;

my $file = './Tests/OMP4/nvprofCloverleafOMP4EventsAndMetrics.csv';
open my $info, $file or die "Could not open $file: $!";
my $line = <$info>;
my $numberofevents = 6;
my $numberofmetrics = 12;
my $cont = 0;
my $events = "l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss,l1_local_store_miss,l1_local_store_hit";
my $metrics = "l1_cache_global_hit_rate,l1_cache_local_hit_rate,ipc,sm_efficiency_instance,achieved_occupancy,sm_efficiency,warp_execution_efficiency,inst_integer,inst_fp_64,inst_control,local_load_transactions,local_store_transactions";
while($cont < 5)
{
$line = <$info>;
#print $line;
$cont++;
}

#print $line;


my $find = 0;
#$line = <$info>;
my @event = split /,/, $events;
my $eventsoutput = "\"Device\", \"Kernel\",\"Invocations\"";
my $eventsoutputSecondRow = ",,";
$cont = 0;
while($cont < ($numberofevents))
{
 $eventsoutput = "$eventsoutput,\"$event[$cont]\",,";
 $eventsoutputSecondRow = "$eventsoutputSecondRow,\"Min\",\"Max\",\"Avg\"";
 $cont++;
}

print "$eventsoutput\n";
print "$eventsoutputSecondRow\n";

while( my $line = <$info>)  {
	my $newrow;
#print $line;
	if($line eq "\n"){
	 $newrow = $line;
	 chomp $newrow;
	 print "$newrow\n";
my @metric = split /,/, $metrics;
my $metricsoutput = "\"Device\", \"Kernel\",\"Invocations\"";
my $metricsoutputSecondRow = ",,";
$cont = 0;
while($cont < ($numberofmetrics))
{
 $metricsoutput = "$metricsoutput,\"$metric[$cont]\",,";
 $metricsoutputSecondRow = "$metricsoutputSecondRow,\"Min\",\"Max\",\"Avg\"";
 $cont++;
}

	 $line = <$info>;
	 $newrow = $line;
	 chomp $newrow;
	 print "$newrow\n";
print "$metricsoutput\n";
print "$metricsoutputSecondRow\n";
         $line = <$info>;
	 $newrow = $line;
	 chomp $newrow;
         $line = <$info>;
#print "find = $find\n";
	 $find = 1;
#print "find = $find\n";
	}
#print "find = $find\n";
if($find == 1){
#found Metrics


chomp $line;	
	
		my @substr = split /,/, $line;
		#print $substr[6];
		my ($correctName) = $substr[1] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;
		$correctName = "" unless $correctName;
		$correctName = "\"$correctName\"";

		#print "correct name $correctName \n";

		
		if ($correctName eq "\"\""){
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
		my ($correctName) = $substr[1] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;
		$correctName = "" unless $correctName;
		$correctName = "\"$correctName\"";

		#print "correct name $correctName \n";

		
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

		 #chomp $newrow;
		}

		
		
		
	}


	print "$newrow\n";

}


