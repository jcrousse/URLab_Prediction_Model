function render_chart(selected_year){

    var width = 720,
        height = 525,
        radius = (Math.min(width, height) / 2) - 10;

    var formatNumber = d3.format(",d");

    var current_data = 0; // 0 = full, 1 = train, 2 = test.

    var x = d3.scaleLinear()
        .range([0, 2 * Math.PI]);

    var y = d3.scaleSqrt()
        .range([0, radius]);

    var color = d3.scaleOrdinal(d3.schemeCategory20);

    var partition = d3.partition();

    var arc = d3.arc()
        .startAngle(function(d) { return Math.max(0, Math.min(2 * Math.PI, x(d.x0))); })
        .endAngle(function(d) { return Math.max(0, Math.min(2 * Math.PI, x(d.x1))); })
        .innerRadius(function(d) { return Math.max(0, y(d.y0)); })
        .outerRadius(function(d) { return Math.max(0, y(d.y1)); });


    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height)
      .append("g")
        .attr("transform", "translate(" + width / 2 + "," + (height / 2) + ")");

    d3.json("sunburst_" + selected_year + ".json", function(error, root) {
      if (error) throw error;

      root = d3.hierarchy(root);
      root.sum(function(d) { return d.size; });

      var g = svg.selectAll("g")
       .data(partition(root).descendants())
       .enter().append("g")
       .attr("id", function(d) { return d.data.name ; })

      //svg.selectAll("path")
          //.data(partition(root).descendants())
        //.enter().append("path")
      var path = g.append("path")
          .attr("d", arc)
          .style("fill", function(d) { return d.data.color; })
          .on("click", toggle_data_selection)
        .append("title")
          .text(function(d) { return d.data.name ; });

      /*var text = g.append("text")
        .attr("transform", function(d) { return "rotate(" + d.data.name === "flare" ? 0 : computeTextRotation(d) + ")"; })
        .attr("x", function(d) { return y(d.y0); })
        .attr("dx", "6") // margin
        .attr("dy", ".35em") // vertical-align
        .text(function(d) { return d.data.name === "flare" ?  d.data.name : "" });*/

      var text = d3.select("#flare"+selected_year).append("text");

      text.append('tspan')
        .attr("text-anchor", "middle")
        .attr("font-family", 'verdana')
        .attr("font-size", 35)
        .text("Year " + selected_year);
      text.append('tspan')
        .attr("text-anchor", "middle")
        .attr("font-family", 'verdana')
        .attr("font-size", 20)
        .text('ROC curve:')
        .attr('x', '.0em')
        .attr('dy', '2.0em');
      text.append('tspan')
        .attr("text-anchor", "middle")
        .attr("font-family", 'verdana')
        .attr("id", "ROC-Value")
        .attr("font-size", 35)
        .text('90%')
        .attr('x', '.0em')
        .attr('dy', '1.0em');
      text.append('tspan')
        .attr("text-anchor", "middle")
        .attr("font-family", 'verdana')
        .attr("id", "data_set")
        .attr("font-size", 15)
        .text('Full dataset')
        .attr('x', '.0em')
        .attr('dy', '-8.0em');
    });

    function toggle_data_selection() {

    current_data = (current_data + 1) % 3;
    if(current_data ==0){ // All data
        d3.selectAll("path").transition().duration(750).style("fill", function(d) { return d.data.color; })
        d3.select("#ROC-Value").text('90%');
        d3.selectAll("#data_set").text('Full dataset');
        }
    if(current_data ==1){ // Train set
        d3.selectAll("path").transition().duration(750).style("fill", function(d) { return  color_pick(d, 1);})
        d3.select("#ROC-Value").text('80%');
        d3.selectAll("#data_set").text('Train set');
        }
    if(current_data ==2){ // Test set
        d3.selectAll("path").transition().duration(750).style("fill", function(d) { return  color_pick(d, 0);})
        d3.select("#ROC-Value").text('75%');
        d3.selectAll("#data_set").text('test set');
        }
    }

    function color_pick(d, select_train_set){
        var color = d.data.color;
        var is_train_set = -1;
        if(d.data.train_set === undefined){
            test = "stop"
        }
        var obj_name = d.data.name.substring(0,5)
        if(obj_name != "flare" && typeof d.data.train_set != "undefined"){
            is_train_set = d.data.train_set
        }else{
            if(obj_name == "flare"){
                stop_here = d.data.name
                }
            if(obj_name != "flare"){
                if(d.parent == null){
                    console.log(d);
                    console.log(d.data.name );}
                is_train_set = d.parent.data.train_set
            }}
        if((is_train_set  == 1 && select_train_set == 0)||
        (is_train_set  == 0 && select_train_set == 1)){
            color = "#ffffff";
         }

         return color
    }

    function click(d) {
      svg.transition()
          .duration(750)
          .tween("scale", function() {
            var xd = d3.interpolate(x.domain(), [d.x0, d.x1]),
                yd = d3.interpolate(y.domain(), [d.y0, 1]),
                yr = d3.interpolate(y.range(), [d.y0 ? 20 : 0, radius]);
            return function(t) { x.domain(xd(t)); y.domain(yd(t)).range(yr(t)); };
          })
        .selectAll("path")
          .attrTween("d", function(d) { return function() { return arc(d); }; });
    }

    function computeTextRotation(d) {
      return (x((d.x0 + d.x1)/2) - Math.PI / 2) / Math.PI * 180;
    }

    d3.select(self.frameElement).style("height", height + "px");
}
render_chart(2014)
render_chart(2015)
render_chart(2016)
render_chart(2017)