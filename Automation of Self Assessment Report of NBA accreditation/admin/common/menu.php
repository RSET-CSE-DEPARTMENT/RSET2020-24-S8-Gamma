<?php
session_start();
$title="";
?><!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8" />
	<link rel="icon" type="image/png" href="assets/img/favicon.ico">
	<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />

	<title></title>

	<meta content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0' name='viewport' />
    <meta name="viewport" content="width=device-width" />


    <!-- Bootstrap core CSS     -->
    <link href="../assets/css/bootstrap.min.css" rel="stylesheet" />

    <!-- Animation library for notifications   -->
    <link href="../assets/css/animate.min.css" rel="stylesheet"/>

    <!--  Light Bootstrap Table core CSS    -->
    <link href="../assets/css/light-bootstrap-dashboard.css" rel="stylesheet"/>


    <!--  CSS for Demo Purpose, don't include it in your project     -->
    <link href="../assets/css/demo.css" rel="stylesheet" />


    <!--     Fonts and icons     -->
    <link href="../common/font-awesome.min.css" rel="stylesheet">
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,700,300' rel='stylesheet' type='text/css'>
    <link href="../assets/css/pe-icon-7-stroke.css" rel="stylesheet" />

</head>
<body>

<div class="wrapper">

<div class="sidebar" data-color="blue" data-image="../assets/img/sidebar-4.jpg">

    <!--

        Tip 1: you can change the color of the sidebar using: data-color="blue | azure | green | orange | red | purple"
        Tip 2: you can also add an image using data-image tag

    -->

    	<div class="sidebar-wrapper">
            <div class="logo">
                
                  <span style="font-size:20px;color:#FFF;margin-top:12px;"> Welcome <br><span style="font-size:12px;color:#000;"> 
                </span>
                </span>
               
            </div>

            <ul class="nav">
                
                
                <li class="active">
                    <a href="../dashboard/dashboard.php">
                        <i class="pe-7s-graph"></i>
                        <p>Home</p>
                    </a>
                </li>
                
                <li>
                    <a href="../registration/select.php">
                        <i class="pe-7s-user"></i>
                        <p>Users</p>
                    </a>
                </li>
                
                
                
                
                
                
                
              <!--  <li>
                    <a href="../customer/select.php">
                        <i class="pe-7s-news-paper"></i>
                        <p>Client</p>
                    </a>
                </li>
                -->
                
                 <li>
                    <a href="../post/select.php">
                        <i class="pe-7s-news-paper"></i>
                        <p>Post</p>
                    </a>
                </li>
                
                 <li>
                    <a href="../wall/select.php">
                        <i class="pe-7s-news-paper"></i>
                        <p>Wall </p>
                    </a>
                </li>
                
             
                
              
                 <li>
                    <a href="../../Social/login.php">
                        <i class="pe-7s-bell"></i>
                        <p>Log Out</p>
                    </a>
                </li>
               
				
            </ul>
    	</div>
    </div>
    <div class="main-panel">
    
    
    
    <nav class="navbar navbar-default navbar-fixed">
            <div class="container-fluid">
                <div class="navbar-header">
                    <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#navigation-example-2">
                        <span class="sr-only">Toggle navigation</span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </button>
                    <br>
                    <?php //echo $_SESSION['company_name']; ?>
                </div>
                <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-left">
                        <li>
                            
                               
								<p class="hidden-lg hidden-md"> <?php echo $_SESSION['company_name']; ?></p>
                            
                        </li>
                        <!--<li class="dropdown">
                              <a href="#" class="dropdown-toggle" data-toggle="dropdown">
                                    <i class="fa fa-globe"></i>
                                    <b class="caret hidden-sm hidden-xs"></b>
                                    <span class="notification hidden-sm hidden-xs">5</span>
									<p class="hidden-lg hidden-md">
										5 Notifications
										<b class="caret"></b>
									</p>
                              </a>
                              <ul class="dropdown-menu">
                                <li><a href="#">Notification 1</a></li>
                                <li><a href="#">Notification 2</a></li>
                                <li><a href="#">Notification 3</a></li>
                                <li><a href="#">Notification 4</a></li>
                                <li><a href="#">Another notification</a></li>
                              </ul>
                        </li>-->
                    <!--    <li>
                           <a href="">
                                <i class="fa fa-search"></i>
								<p class="hidden-lg hidden-md">Search</p>
                            </a>
                        </li>-->
                    </ul>

                    <ul class="nav navbar-nav navbar-right">
                      
                       
                        <li>
                            <a href="../../Social/login.php">
                                <p>Log out</p>
                            </a>
                        </li>
						<li class="separator hidden-lg hidden-md"></li>
                    </ul>
                </div>
            </div>
        </nav>