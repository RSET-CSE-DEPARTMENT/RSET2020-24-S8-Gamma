 <script type="text/javascript">     
        function PrintDiv() {    
           var divToPrint = document.getElementById('divToPrint');
           var popupWin = window.open('', '_blank', 'width=300,height=300');
           popupWin.document.open();
           popupWin.document.write('<html><body onload="window.print()">' + divToPrint.innerHTML + '</html>');
            popupWin.document.close();
                }
     </script>
<div id="divToPrint">
<?php
error_reporting(0);

$con=mysqli_connect("localhost","root","","sar");
	echo "<table border=0 style='border-collapse:collapse;' width='1000' align='center'>";
for($i=0;$i<9;$i++)
{

$query=mysqli_query($con,"SELECT * FROM master_question where criteria='$i' order by criteria_position asc") or die(mysql_error()); 
 while($row = mysqli_fetch_array($query))
  {
	  if($row['table_name']=="")
	  {
	  echo "<tr><td valign='top' width='30px'> $i:$row[criteria_position]</td><td> <b>$row[question]</b> </td> </tr>";
	  
	  
	  $query2=mysqli_query($con,"SELECT * FROM master_answer where question_id='$row[id]'") or die(mysql_error()); 
 while($row2 = mysqli_fetch_array($query2))
  {
	 echo "<tr><td > </td><td style='text-align:justify;'> $row2[answer] ";
	 
	 if($row2['document']!="")
	 $dd=explode(".",$row2['document']);
 if($dd[1]!="")
 {
	 if($dd[1]=="png" || $dd[1]=="jpg" || $dd[1]=="jpeg")
	 echo "<img src='master_answer/uploads/$row2[document]' width='600'>";
	  if($dd[1]=="pdf" || $dd[1]=="xlsx" || $dd[1]=="doc")
	 echo "<br><a href='master_answer/uploads/$row2[document]' target='_blank'><b>Download</b> </a>";
	
	 
 }
 
  if($row2['answer']!="")
  {
		 echo "<br><b>Mark = $row[marks]</b>";
		 $total=$total+$row['marks'];
  }
	 echo "</td> </tr>"; 
  }
	  	 echo "<tr><td > </td><td>  <br><br><br><br></td> </tr>"; 
	 
	  }
  else
  {
	  $table=$row['table_name'];
	  
	  	  echo "<tr><td colspan=2> <br><br>";

	 include("stb.php");
	  
	if($flagg==1)
	{
		 echo "<br><b>Mark = $row[marks]</b>";
		 $total=$total+$row['marks'];
	}
	 $flagg=0;
	  
	  echo "
		 <br><br> </td> </tr>";
	  
	  
  }
}
	
}
echo "</table>";
echo "<h2>Total : $total</h2>";
?></div>

	 <center>
    <input type="submit" value="Print" onClick="PrintDiv();" class='btn btn-primary'/>
    </center>