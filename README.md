Tensorflow Environment Setup
============================

* Install docker-machine & virtualbox
	- brew install docker-machine
	- install virtualbox by click [here](http://download.virtualbox.org/virtualbox/5.1.8/VirtualBox-5.1.8-111374-OSX.dmg)

* Create virtual machine
	- docker-machine create -d virtualbox --virtualbox-memory 8196 tensorflow

* Look the ip of the virtual machine
	- docker-machine ip tensorflow

* Add this command into you environment file
	- eval $(docker-machine env tensorflow)

* Running the Docker container from the TW AI Club repository
	- docker run -p 8888:8888 --name tensorflow -d twaiclub/tensorflow-python3

* Go to: http://your-virtual-machine-ip:8888, and then you can do everything you want

* You also can use other ways to setup environment, just reminder please provide more memory as the calculation need.

* And there is a link about this course in [udacity](https://classroom.udacity.com/courses/ud730). You can get more information from it.