#!/usr/bin/perl

use strict;
use warnings;

my $file = './Tests/OMP4/nvprofCloverleafOMP4EventsAndMetrics.csv';
open my $info, $file or die "Could not open $file: $!";
my $line = <$info>;

my $cont = 0;
while($cont < 5){
$line = <$info>;
#print $line;
$cont++;
}

#print $line;


my $trovato = 0;

while( my $line = <$info>)  {
	my $newrow;
	if($line =~ /==[a-zA-Z0-9_.-]*== API /g)
	{
	  $trovato = 1;
	}

	if($trovato == 1 || $line eq "\n"){
	 $newrow = $line;
	 chomp $newrow;
	 print "$newrow\n";
	 $line = <$info>;
	 $newrow = $line;
	 chomp $newrow;
	 print "$newrow\n";
         $line = <$info>;
	 $newrow = $line;
	 chomp $newrow;
         $line = <$info>;
	}else{
	
		#print $line;
	
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
		 $newrow = "$substr[0],$correctName,$substr[2],$substr[3],$substr[4],$substr[5],$substr[6]";
		 chomp $newrow;
		}
	}

	print "$newrow\n";

}

