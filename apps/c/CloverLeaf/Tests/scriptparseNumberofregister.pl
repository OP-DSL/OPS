#!/usr/bin/perl

use strict;
use warnings;
my $numberofrowbetweenkernels=5;
my $rnkernelname = 1;
my $rnstackframeandco = 3;
my $rnnumberofregisters = 4;
#my $file = './Tests/OMP4/outnumberofregistersforkernel';

my $file = $ARGV[0];
open my $info, $file or die "Could not open $file: $!";
my $line = <$info>;
while( my $line = <$info>)  {   
    
    if($line =~ "gmem")
    {
	 	#print $line; 
	my $row;

	my $cont = 0;
        my $myrow;
	while($cont < $numberofrowbetweenkernels){
                $row = $line;
		chomp $row;
		

		if($cont == $rnkernelname){
			my (@substr) = $row =~ m/\'(.*?)\'/g;
			#print "@substr\n";
my ($substr2) = $substr[0] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;
                        $myrow = "$substr2;$substr[1];";

		}

                if($cont == $rnstackframeandco){
                        my @substr = split /,/, $row;
			#print "@substr\n";
			$myrow .= "$substr[0];$substr[1];$substr[2];";

		}

                if($cont == $rnnumberofregisters){
                        my (@substr) = $row =~ m/ptxas info    \: Used(.*?) registers/g;
			#print "@substr\n";
			$myrow .= "$substr[0]";

		}

		
		$row = "$row;$line";
		$line = <$info>;
		$cont++;
        }
     	print "$myrow\n";
    }

}

close $info;
