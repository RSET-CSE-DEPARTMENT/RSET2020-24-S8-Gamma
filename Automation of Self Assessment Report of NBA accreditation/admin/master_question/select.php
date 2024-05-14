<?php
include("../header_inner.php");
include("config.php");

$del_id=0;
$i=0;
?>


		<link rel="stylesheet" type="text/css" href="datatables.min.css">
 
		<script type="text/javascript" src="datatables.min.js"></script>
		<script type="text/javascript" charset="utf-8">
			$(document).ready(function() {
				$('#example').DataTable();
			} );
		</script>

<style>
.hiddentd
{
display:inline-block;
    width:180px;
    white-space: nowrap;
    overflow:hidden !important;
   
}
</style>


<div class="">
<?php

	echo "<div class='col-sm-2' style='float:right;margin-bottom:10px;'><form action='form.php' method='post'><input type='submit' name='view' value='Add New' class='form-control btn-danger'></form></div>";
	
?>
<div class="clearfix"></div>
<table id="example" class="table table-striped table-bordered dataTable no-footer" cellspacing="0"  role="grid" aria-describedby="example_info" >

       
        
            
          <?php
	
		  include("../connection.php");
	
	
	
	
	
	
	
	
if(isset($_REQUEST['del_id']))
{
$del_id=$_REQUEST['del_id'];
mysqli_query($con,"delete from $table where id='$del_id'");
//if($del_id!="")
}
if(isset($_REQUEST['u_id']))
{
$u_id=$_REQUEST['u_id'];
mysqli_query($con,"UPDATE $table status='Block' where id='$u_id'");
//if($del_id!="")
}
	?>
    <script>


function rem()
{
if(confirm('Are you sure you want to delete this record?')){
return true;
}
else
{
return false;
}
}


function rem2()
{
if(confirm('Are you sure you want to deactive this record?')){
return true;
}
else
{
return false;
}
}
</script>
    
	
	<?php


	
	
	

	
	
		  $result2 = mysqli_query($con,"SHOW FIELDS FROM $table");

 echo "<thead><tr>";

while ($row2 = mysqli_fetch_array($result2))
 {

  $name=$row2['Field'];
if($i==10)
{
}
elseif($i==999)
{
}
elseif($i==190)
{
}
else
{
  echo "<th>".
  str_replace('_', ' ', $name)
  ."</th>";
}
 $i++;
 }
 echo "

 <th>Del</th> 
 </tr></thead>";
   
  // $i=0;
   echo "<tbody>";
   
            
            
         
 	$result = mysqli_query($con,"SELECT * FROM $table ");
	

		while($row = mysqli_fetch_array($result))
		{
		$id=$row['id'];
		
		echo "<tr>";
		for($k=0;$k<$i;$k++)
		{
	
			
			if($k==60)
			{
			  $sql2 = "select *  from customer where id='$_SESSION[userid]' ";
    $result2 = mysqli_query($con, $sql2) or die("Error in Selecting " . mysqli_error($connection));
$row2 =mysqli_fetch_array($result2);
		
		

			echo "<td >  $row2[contact_person]</td>";
				
			}
			
			elseif($k==10 )
			{
				
				
			}
			elseif($k==9)
			{
				
				
			}
			elseif($k==28)
			{
				
				
			}
			
				
			elseif($k==30)
			{
				

			echo "<td class='hiddentd'> $row[content] </td>";
				
			}
			
			
				elseif($k==11)
			{
			  

			echo "<td > <img src='../../Social/$row[$k]' width='100'></td>";
				
			}
				elseif($k==12)
			{
			  
			  if($z>5)
			  {
				  
			echo "<td>User Blocked </td>";
			  }
			  else{
				  
				echo "<td>$row[$k]</td>";
			  }

				
			}
			
			else
			{
			echo "<td >$row[$k]</td>";
			}
		
		
		
		
		
		}
		
		
		
		
			echo "
			
			<td>
			
			<a href='../master_answer/form.php?qid=$id' target='_blank'>Add Answer</a>
			<a href='update.php?id=$id'>Update</a>
			<a href='?del_id=$id' onclick='return rem()' class='btn btn-danger col-sm-12'>Del</a></td>
		
			</tr>";
		
		
		
		}
        
        ?>
        </tbody>
    </table>
			
		



<script type="text/javascript">
	// For demo to fit into DataTables site builder...
	$('#example')
		.removeClass( 'display' )
		.addClass('table table-striped table-bordered');
</script>

<div class="clearfix"></div>
	
    </div> 
    <?php
	
//	include("../footer_inner.php");
	?>