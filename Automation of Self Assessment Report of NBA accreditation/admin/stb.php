



<table border=1 style='border-collapse:collapse;' width='1000' align='center'>

       
        
    
	
	<?php


	
	
	$ii=0;

	
	
$result2 = mysqli_query($con,"SHOW FIELDS FROM $table");

 echo "<thead><tr>";

while ($row2 = mysqli_fetch_array($result2))
 {

  $name=$row2['Field'];

  echo "<th>".
  str_replace('_', ' ', $name)
  ."</th>";

 $ii++;
 }
 echo "


 </tr></thead>";
   
  // $i=0;
   echo "<tbody>";
   
            
            
         
 	$result3 = mysqli_query($con,"SELECT * FROM $table ");
	

		while($row3 = mysqli_fetch_array($result3))
		{
		$id=$row3['id'];
		$flagg=1;
		echo "<tr>";
		for($k=0;$k<$ii;$k++)
		{
	
		
			echo "<td >$row3[$k]</td>";
	
		
		
		
		
		
		}
		
		
		
		
		
		
		
		}
        
        ?>
        </tbody>
    </table>
			
<?php

if($table=="cr41table")
{
$query2323=mysqli_query($con,"SELECT * FROM cr41table where id='3' ") or die(mysql_error()); 
 while($row2323 = mysqli_fetch_array($query2323))
  {
	  $avg=($row2323['cay']+$row2323['caym1']+$row2323['caym2'])/3;
	  $flagg=0;
	  echo "sssssssssssssss".$avg;
  }
  if($avg>80)
   $total=$total+5;
elseif($avg>60)
$total=$total+3;
}

?>