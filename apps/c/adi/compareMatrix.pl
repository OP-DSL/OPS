#! /usr/bin/perl
#use strict;
use warnings;
use File::Compare;
use File::Basename;

sub ltrim { my $s = shift; $s =~ s/^\s+//;       return $s };
sub rtrim { my $s = shift; $s =~ s/\s+$//;       return $s };
sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

$ARGV[0] || die "*PATH matrix 1 not found*\n";
$ARGV[1] || die "*PATH matrix 2 not found*\n";

$PATH1 = $ARGV[0];
$PATH2 = $ARGV[1];

$TOLERANCE = $ARGV[2];

unless ( -f $PATH1 ) { die "File 1 Doesn't Exist!\n"; }
unless ( -f $PATH2 ) { die "File 2 Doesn't Exist!\n"; }

if ( "$PATH1" eq "$PATH2" ) {
    die "Comparing the same files!\nExiting";
}
else {

    open( FILE1, $ARGV[0] );
    open( FILE2, $ARGV[1] );
    $count    = 0;
    $errorSum = 0.0;
    $max      = 0.0;
    $r        = -1.0;
    $c        = -1.0;

    $cumulative_sum1 = 0.0;
    $cumulative_sum2 = 0.0;

    while ( $line1 = <FILE1> ) {
        $line1 = trim($line1); 

        $c           = 0;
        $r           = $r + 1;
        $line2       = <FILE2>;
        $line2 = trim($line2);
        @valuesLine1 = split( / /, $line1 );
        @valuesLine2 = split( / /, $line2 );
        $size        = @valuesLine1;
        for ( $i = 0 ; $i < $size ; $i++ ) {
            $c = $c + 1;

            #remove trailing new line
            #$valuesLine1[$i] = trim ($valuesLine1[$i] );
            #$valuesLine2[$i] = trim ($valuesLine2[$i] );

            #compute comulative sum of both files (might overflow!)
            $cumulative_sum1 +=  $valuesLine1[$i];
            $cumulative_sum2 +=  $valuesLine2[$i];
            if ( $valuesLine1[$i] != $valuesLine2[$i] ) {
                $errorSum += abs( $valuesLine1[$i] - $valuesLine2[$i] );
                $count++;
                $abs_difference = abs( $valuesLine1[$i] - $valuesLine2[$i] );
                if ( $abs_difference > $max ) {
                    $max = $abs_difference;
                }
            }
        }
    }
    $avgError = 0;
    if ( $count > 0 ) {
        $avgError = $errorSum / $count;
    }
    if ( $max <= $TOLERANCE ) {

        #print ok if the the max error is below the tolerance
        print "OK !";
    }

    #pretty print some info
    print "\n";
    print_n_times( '-', 99 );
    print "\n";
    printf(
        "| %-30s | %-30s | %-30s|\n",
        "DIFF CELL COUNT",
        "AVG ERROR", "MAX ERROR"
    );
    printf( "| %-30s | %-30s | %-30s|\n", $count, $avgError, $max );
    print_n_times( '-', 99 );
    print "\n";

    my $cuumulative_sum_difference = abs($cumulative_sum1-$cumulative_sum2);
    if( $cuumulative_sum_difference >= $TOLERANCE){
        printf("CUMULATIVE SUMS NOT MATCHING!");
    }
    printf("CUMULATIVE SUM (%s) -> %.20f\n",$PATH1,$cumulative_sum1);
    printf("CUMULATIVE SUM (%s) -> %.20f\n",$PATH2,$cumulative_sum2);
    printf("CUMULATIVE SUM DIFFERENCE -> %.20f\n",$cuumulative_sum_difference);

    close(FILE1);
    close(FILE2);

    print "\n";
}

sub print_n_times {
    my ( $char, $times ) = @_;
    for ( my $i = 0 ; $i < $times ; $i++ ) {
        print "$char";
    }
}
