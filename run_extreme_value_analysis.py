# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:37:19 2023

@author: admin
"""

import pyextremes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_eva(df, tag, block_size=pd.Timedelta(pd.to_timedelta("365.2425D")),
            csize=15, minima=False):
    
    """
    run the extreme value analysis and create
    
    1) a plot of the extremes in a timeseries
    2) a summary plot of model fitting and return values
    
    inputs
    df - dataframe with datetime as index
    tag - name of dataframe column to do EVA on
    block_size - time length to chop data up into
    csize - fontsize for plotting
    minima - bool, whether to look for minima (if True, looking for negative peaks)
    
    returns
    gevd_fit_params - df containing shape_, location, scale, and distribution_name
    extremes_df - df containing datetime and value of the detected extremes
    fig_extremes - fig object showing timeseries of tag, with extremes highlighted
    ax_extremes - ax object for fig_extremes
    fig_model - fig object assessing model fit
    ax_model - ax object for fig_model
    summary - df of return values and their probabilities
    """
    
    
    extremes_type='low' if minima==True else 'high'

    # Initialise model
    eva_model=pyextremes.EVA(df[tag])
    eva_model.get_extremes("BM", block_size=block_size,extremes_type=extremes_type)
    eva_model.fit_model(model="Emcee")  
    
    # Extract fit parameters
    gevd_fit_params=pd.DataFrame(eva_model.model.fit_parameters, index=[0])
    gevd_fit_params.rename(columns={'c':'shape_', 'loc':'location'}, inplace=True)
    
    if eva_model.distribution.name == 'gumbel_r':
        gevd_fit_params['shape_']=0.0
        location_quantiles=np.quantile(eva_model.model.trace[:,:,0].flatten(), [0.025, 0.975])
        scale_quantiles=np.quantile(eva_model.model.trace[:,:,1].flatten(), [0.025, 0.975])
        
        gevd_fit_params['shape_lower_ci_width']=np.nan
        gevd_fit_params['shape_upper_ci_width']=np.nan
        
        gevd_fit_params['location_lower_ci_width']=gevd_fit_params.location-location_quantiles[0]
        gevd_fit_params['location_upper_ci_width']=location_quantiles[1]-gevd_fit_params.location
        
        gevd_fit_params['scale_lower_ci_width']=gevd_fit_params.scale-scale_quantiles[0]
        gevd_fit_params['scale_upper_ci_width']=scale_quantiles[1]-gevd_fit_params.scale

    else:
        # Calculate the 95% confidence intervals on fit params
        shape_quantiles=np.quantile(eva_model.model.trace[:,:,0].flatten(), [0.025, 0.975])
        location_quantiles=np.quantile(eva_model.model.trace[:,:,1].flatten(), [0.025, 0.975])
        scale_quantiles=np.quantile(eva_model.model.trace[:,:,2].flatten(), [0.025, 0.975])
        
        gevd_fit_params['shape_lower_ci_width']=gevd_fit_params.shape_-shape_quantiles[0]
        gevd_fit_params['shape_upper_ci_width']=shape_quantiles[1]-gevd_fit_params.shape_
        
        gevd_fit_params['location_lower_ci_width']=gevd_fit_params.location-location_quantiles[0]
        gevd_fit_params['location_upper_ci_width']=location_quantiles[1]-gevd_fit_params.location
        
        gevd_fit_params['scale_lower_ci_width']=gevd_fit_params.scale-scale_quantiles[0]
        gevd_fit_params['scale_upper_ci_width']=scale_quantiles[1]-gevd_fit_params.scale
    
    gevd_fit_params['distribution_name']= eva_model.distribution.name   
    
        
    # Extract extremes
    extremes_df=pd.DataFrame({'datetime':eva_model.extremes.index, tag:eva_model.extremes.values})
    
    # Plot extremes
    fig_extremes,ax_extremes=plt.subplots(figsize=(15,10))
    ax_extremes.plot(df[tag], color='grey', linewidth=1.0, label=str(tag))
    ax_extremes.plot(extremes_df.datetime, extremes_df[tag], color='mediumorchid', linewidth=0.0, label='extremes', marker='*', markersize=15)
    

    ax_extremes.set_xlabel('Year', fontsize=csize)
    ax_extremes.set_ylabel(str(tag), fontsize=csize)
    for label in (ax_extremes.get_xticklabels() + ax_extremes.get_yticklabels()):
        label.set_fontsize(csize) 
    ax_extremes.legend(fontsize=csize)


    # Overall plot for assessing the model
    fig_model, ax_model=plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    observed_return_values=pyextremes.get_return_periods(ts=eva_model.data, extremes=eva_model.extremes, 
            extremes_method=eva_model.extremes_method, extremes_type=eva_model.extremes_type,
            block_size=eva_model.extremes_kwargs.get("block_size", None), return_period_size='365.2425D' )
    return_period=np.linspace(observed_return_values.loc[:, "return period"].min(),
            observed_return_values.loc[:, "return period"].max(),100,)    
    modeled_return_values = eva_model.get_summary(return_period=return_period, return_period_size='365.2425D',alpha=0.95)

    ax_model[0,0].plot(observed_return_values['return period'], observed_return_values[tag], linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')
    ax_model[0,0].plot(modeled_return_values.index, modeled_return_values['return value'], linewidth=2.0, color='coral', label='Model')
    ax_model[0,0].fill_between(modeled_return_values.index, modeled_return_values['lower ci'], modeled_return_values['upper ci'], color='grey', alpha=0.5, label='95% CI')
    
    ax_model[0,0].set_xlabel('Return Period (years)', fontsize=csize)
    ax_model[0,0].set_ylabel(str(tag)+' observed at least once\nper return period (nT)', fontsize=csize)
    ax_model[0,0].set_xscale('log')
    ax_model[0,0].legend(fontsize=csize, loc='lower right')  
    
    for label in (ax_model[0,0].get_xticklabels() + ax_model[0,0].get_yticklabels()):
        label.set_fontsize(csize) 
      
    t=ax_model[0,0].text(0.06,0.94,'(a)', transform=ax_model[0,0].transAxes, fontsize=csize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    # Compare the distributions
    ax_model[0,1].hist(eva_model.extremes.values,  density=True, rwidth=0.8, color='grey', label='Observations')

    pdf_support = np.linspace(eva_model.extremes.min(), eva_model.extremes.max(), 100)
    pdf = eva_model.model.pdf(eva_model.extremes_transformer.transform(pdf_support))

    ax_model[0,1].plot(pdf_support, pdf, color='coral', label='Model')
    
    ax_model[0,1].set_xlabel('Extremes - '+str(tag), fontsize=csize)
    ax_model[0,1].set_ylabel('Normalised Occurrence', fontsize=csize)
    
    ax_model[0,1].legend(fontsize=csize)
    
    for label in (ax_model[0,1].get_xticklabels() + ax_model[0,1].get_yticklabels()):
        label.set_fontsize(csize)

    t=ax_model[0,1].text(0.06,0.94,'(b)', transform=ax_model[0,1].transAxes, fontsize=csize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    #params_text=r'$\mu$ = '+str(float('%.4g' % gevd_fit_params.location))+'\n$\sigma$ = '+str(float('%.4g' % gevd_fit_params.scale))+'\n'+r'$\xi$ = '+str(float('%.4g' % gevd_fit_params.shape_))
    params_text=r'$\mu$ = '+str(float('%.4g' % gevd_fit_params.location)) + ' (-' +str(float('%.4g' % gevd_fit_params.location_lower_ci_width)) +', +'+ str(float('%.4g' % gevd_fit_params.location_upper_ci_width)) +')'+'\n$\sigma$ = '+str(float('%.4g' % gevd_fit_params.scale)) + ' (-' +str(float('%.4g' % gevd_fit_params.scale_lower_ci_width)) +', +'+ str(float('%.4g' % gevd_fit_params.scale_upper_ci_width)) +')'+'\n'+r'$\xi$ = '+str(float('%.4g' % gevd_fit_params.shape_)) + ' (-' +str(float('%.4g' % gevd_fit_params.shape_lower_ci_width)) +', +'+ str(float('%.4g' % gevd_fit_params.shape_upper_ci_width)) +')'
    t_m=ax_model[0,1].text(0.94,0.75,params_text,transform=ax_model[0,1].transAxes, fontsize=csize, va='top', ha='right' )



    # # Plot a q-q plot
    observed = observed_return_values.loc[:, eva_model.extremes.name].values
    theoretical = eva_model.extremes_transformer.transform(eva_model.model.isf(observed_return_values.loc[:, "exceedance probability"].values))

    # Observed is just the observed extreme B values
    # Theoretical takes the probability for the observed B, and then extracts the predicted B from the model

    #fig,ax=plt.subplots()
    ax_model[1,0].plot(theoretical, observed, linewidth=0.0, marker='o', fillstyle='none', color='mediumslateblue')
    
    min_value = min([min(ax_model[1,0].get_xlim()), min(ax_model[1,0].get_ylim())])
    max_value = max([max(ax_model[1,0].get_xlim()), max(ax_model[1,0].get_ylim())])
    ax_model[1,0].plot( [min_value, max_value], [min_value, max_value], linewidth=1.0, linestyle='--', color='black')
    
    ax_model[1,0].set_xlabel('Model '+str(tag), fontsize=csize)
    ax_model[1,0].set_ylabel('Observed '+str(tag), fontsize=csize)
    
    ax_model[1,0].set_title('QQ plot', fontsize=csize)  
    
    for label in (ax_model[1,0].get_xticklabels() + ax_model[1,0].get_yticklabels()):
        label.set_fontsize(csize)

    t=ax_model[1,0].text(0.06,0.94,'(c)', transform=ax_model[1,0].transAxes, fontsize=csize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    fig_model.tight_layout()


    # Plot a table of return values
    summary = eva_model.get_summary(
        return_period=[2, 5, 10,15,20, 25, 50, 100],
        alpha=0.95 )
    summary=summary.reset_index()

    # Format the DF for the table
    summary=summary.round()
    summary=summary.rename(columns={"return period": "period",
                            "return value": "value",
                            "lower ci": "-95% CI",
                            "upper ci": "+95% CI"})

    summary_new=pd.DataFrame({"period":summary['period'],
                              "value":summary['value'],
                              "-95% CI":summary['value'] - summary['-95% CI'],
                              "+95% CI":summary['+95% CI'] - summary['value']
                              })


    table=ax_model[1,1].table(cellText=summary_new.values, colLabels=summary_new.columns, loc='center')#, fontsize=csize+2)
    table.auto_set_font_size(False) # stop auto font size
    table.set_fontsize(csize)       # increase font size
    table.scale(1,3)    # don't increase cell width (1) but increase height x3
    ax_model[1,1].axis('off')
    t=ax_model[1,1].text(0.06,0.94,'(d)', transform=ax_model[1,1].transAxes, fontsize=csize, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    
    #opng=r'Users\admin\Documents\figures\sc_magnetometer_study\sw_params_extremes\extreme_value_analysis_'+str(tag)+'.png'
    #opng=os.path.join('C:'+os.sep,opng)
    #fig.savefig(opng)

    ## Save return values to file
    #ocsv=r'Users\admin\Documents\figures\sc_magnetometer_study\dbdt_npys\return_values_'+str(tag)+'.csv'
    #ocsv=os.path.join('C:'+os.sep,ocsv)
    #summary.to_csv(ocsv, index=False) 
    
    return gevd_fit_params, extremes_df, fig_extremes, ax_extremes, fig_model, ax_model, summary
