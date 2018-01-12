#!/usr/bin/perl

use strict;
use warnings;

my $file = './Tests/OMP4/nvprofCloverleafOMP4Classic.csv';
open my $info, $file or die "Could not open $file: $!";
my $line = <$info>;

my $cont = 0;
while($cont < 4){
$line = <$info>;
#print $line;
$cont++;
}
my $trovato = 0;

my $totaltime = 0;
while( my $line = <$info>)  {
	my $newrow;
	if($line =~ /==[a-zA-Z0-9_.-]*== API /g)
	{
	  $trovato = 1;
	}

	if($trovato == 1 || $line eq "\n"){
	 $newrow = $line;

	 chomp $newrow;
	}else{
	
		#print $line;
	
		my @substr = split /,/, $line;
		#print $substr[6];
		my ($correctName) = $substr[6] =~ /(?<=_Z[0-9][0-9])[a-zA-Z0-9_.-]*(?=_wrapper)/g;
		$correctName = "" unless $correctName;
		$correctName = "\"$correctName\"";

		#print "correct name $correctName \n";
		
		if ($correctName eq "\"\""){
		 $newrow = $line;
         my @substr = split /,/, $newrow;
	 $totaltime += $substr[1];
		 chomp $newrow;
		}else{
		$totaltime += $substr[1];
		 $newrow = "$substr[0],$substr[1],$substr[2],$substr[3],$substr[4],$substr[5],$correctName";
		 chomp $newrow;
		}
	}

	print "$newrow\n";

}

print STDERR "total time is : $totaltime ms\n";



